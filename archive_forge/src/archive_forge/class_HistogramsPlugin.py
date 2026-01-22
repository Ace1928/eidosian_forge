from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.histogram import metadata
class HistogramsPlugin(base_plugin.TBPlugin):
    """Histograms Plugin for TensorBoard.

    This supports both old-style summaries (created with TensorFlow ops
    that output directly to the `histo` field of the proto) and new-
    style summaries (as created by the
    `tensorboard.plugins.histogram.summary` module).
    """
    plugin_name = metadata.PLUGIN_NAME
    SAMPLE_SIZE = 51

    def __init__(self, context):
        """Instantiates HistogramsPlugin via TensorBoard core.

        Args:
          context: A base_plugin.TBContext instance.
        """
        self._downsample_to = (context.sampling_hints or {}).get(self.plugin_name, _DEFAULT_DOWNSAMPLING)
        self._data_provider = context.data_provider
        self._version_checker = plugin_util._MetadataVersionChecker(data_kind='histogram', latest_known_version=0)

    def get_plugin_apps(self):
        return {'/histograms': self.histograms_route, '/tags': self.tags_route}

    def is_active(self):
        return False

    def index_impl(self, ctx, experiment):
        """Return {runName: {tagName: {displayName: ..., description:
        ...}}}."""
        mapping = self._data_provider.list_tensors(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME)
        result = {run: {} for run in mapping}
        for run, tag_to_content in mapping.items():
            for tag, metadatum in tag_to_content.items():
                description = plugin_util.markdown_to_safe_html(metadatum.description)
                md = metadata.parse_plugin_metadata(metadatum.plugin_content)
                if not self._version_checker.ok(md.version, run, tag):
                    continue
                result[run][tag] = {'displayName': metadatum.display_name, 'description': description}
        return result

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(element_name='tf-histogram-dashboard')

    def histograms_impl(self, ctx, tag, run, experiment, downsample_to=None):
        """Result of the form `(body, mime_type)`.

        At most `downsample_to` events will be returned. If this value is
        `None`, then default downsampling will be performed.

        Raises:
          tensorboard.errors.PublicError: On invalid request.
        """
        sample_count = downsample_to if downsample_to is not None else self._downsample_to
        all_histograms = self._data_provider.read_tensors(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME, downsample=sample_count, run_tag_filter=provider.RunTagFilter(runs=[run], tags=[tag]))
        histograms = all_histograms.get(run, {}).get(tag, None)
        if histograms is None:
            raise errors.NotFoundError('No histogram tag %r for run %r' % (tag, run))
        events = [(e.wall_time, e.step, e.numpy.tolist()) for e in histograms]
        return (events, 'application/json')

    @wrappers.Request.application
    def tags_route(self, request):
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        index = self.index_impl(ctx, experiment=experiment)
        return http_util.Respond(request, index, 'application/json')

    @wrappers.Request.application
    def histograms_route(self, request):
        """Given a tag and single run, return array of histogram values."""
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        tag = request.args.get('tag')
        run = request.args.get('run')
        body, mime_type = self.histograms_impl(ctx, tag, run, experiment=experiment, downsample_to=self.SAMPLE_SIZE)
        return http_util.Respond(request, body, mime_type)