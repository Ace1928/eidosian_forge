import operator
from .. import errors, ui
from ..i18n import gettext
from ..revision import NULL_REVISION
from ..trace import mutter
class RepoFetcher:
    """Pull revisions and texts from one repository to another.

    This should not be used directly, it's essential a object to encapsulate
    the logic in InterRepository.fetch().
    """

    def __init__(self, to_repository, from_repository, last_revision=None, find_ghosts=True, fetch_spec=None):
        """Create a repo fetcher.

        Args:
          last_revision: If set, try to limit to the data this revision
            references.
          fetch_spec: A SearchResult specifying which revisions to fetch.
            If set, this overrides last_revision.
          find_ghosts: If True search the entire history for ghosts.
        """
        self.to_repository = to_repository
        self.from_repository = from_repository
        self.sink = to_repository._get_sink()
        self._last_revision = last_revision
        self._fetch_spec = fetch_spec
        self.find_ghosts = find_ghosts
        with self.from_repository.lock_read():
            mutter('Using fetch logic to copy between %s(%s) and %s(%s)', str(self.from_repository), str(self.from_repository._format), str(self.to_repository), str(self.to_repository._format))
            self.__fetch()

    def __fetch(self):
        """Primary worker function.

        This initialises all the needed variables, and then fetches the
        requested revisions, finally clearing the progress bar.
        """
        self.count_total = 0
        self.file_ids_names = {}
        with ui.ui_factory.nested_progress_bar() as pb:
            pb.show_pct = pb.show_count = False
            pb.update(gettext('Finding revisions'), 0, 2)
            search_result = self._revids_to_fetch()
            mutter('fetching: %s', str(search_result))
            if search_result.is_empty():
                return
            pb.update(gettext('Fetching revisions'), 1, 2)
            self._fetch_everything_for_search(search_result)

    def _fetch_everything_for_search(self, search):
        """Fetch all data for the given set of revisions."""
        if self.from_repository._format.rich_root_data and (not self.to_repository._format.rich_root_data):
            raise errors.IncompatibleRepositories(self.from_repository, self.to_repository, 'different rich-root support')
        with ui.ui_factory.nested_progress_bar() as pb:
            pb.update('Get stream source')
            source = self.from_repository._get_source(self.to_repository._format)
            stream = source.get_stream(search)
            from_format = self.from_repository._format
            pb.update('Inserting stream')
            resume_tokens, missing_keys = self.sink.insert_stream(stream, from_format, [])
            if missing_keys:
                pb.update('Missing keys')
                stream = source.get_stream_for_missing_keys(missing_keys)
                pb.update('Inserting missing keys')
                resume_tokens, missing_keys = self.sink.insert_stream(stream, from_format, resume_tokens)
            if missing_keys:
                raise AssertionError('second push failed to complete a fetch {!r}.'.format(missing_keys))
            if resume_tokens:
                raise AssertionError('second push failed to commit the fetch {!r}.'.format(resume_tokens))
            pb.update('Finishing stream')
            self.sink.finished()

    def _revids_to_fetch(self):
        """Determines the exact revisions needed from self.from_repository to
        install self._last_revision in self.to_repository.

        Returns:
          A SearchResult of some sort.  (Possibly a
          PendingAncestryResult, EmptySearchResult, etc.)
        """
        from . import vf_search
        if self._fetch_spec is not None:
            return self._fetch_spec
        elif self._last_revision == NULL_REVISION:
            return vf_search.EmptySearchResult()
        elif self._last_revision is not None:
            return vf_search.NotInOtherForRevs(self.to_repository, self.from_repository, [self._last_revision], find_ghosts=self.find_ghosts).execute()
        else:
            return vf_search.EverythingNotInOther(self.to_repository, self.from_repository, find_ghosts=self.find_ghosts).execute()