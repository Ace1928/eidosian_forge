import argparse
import functools
import gzip
import io
import mimetypes
import posixpath
import zipfile
from werkzeug import utils
from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.util import grpc_util
from tensorboard.util import tb_logging
from tensorboard import version
class CorePluginLoader(base_plugin.TBLoader):
    """CorePlugin factory."""

    def __init__(self, include_debug_info=None):
        self._include_debug_info = include_debug_info

    def define_flags(self, parser):
        """Adds standard TensorBoard CLI flags to parser."""
        parser.add_argument('--logdir', metavar='PATH', type=str, default='', help="Directory where TensorBoard will look to find TensorFlow event files\nthat it can display. TensorBoard will recursively walk the directory\nstructure rooted at logdir, looking for .*tfevents.* files.\n\nA leading tilde will be expanded with the semantics of Python's\nos.expanduser function.\n")
        parser.add_argument('--logdir_spec', metavar='PATH_SPEC', type=str, default='', help='Like `--logdir`, but with special interpretation for commas and colons:\ncommas separate multiple runs, where a colon specifies a new name for a\nrun. For example:\n`tensorboard --logdir_spec=name1:/path/to/logs/1,name2:/path/to/logs/2`.\n\nThis flag is discouraged and can usually be avoided. TensorBoard walks\nlog directories recursively; for finer-grained control, prefer using a\nsymlink tree. Some features may not work when using `--logdir_spec`\ninstead of `--logdir`.\n')
        parser.add_argument('--host', metavar='ADDR', type=str, default=None, help='What host to listen to (default: localhost). To serve to the entire local\nnetwork on both IPv4 and IPv6, see `--bind_all`, with which this option is\nmutually exclusive.\n')
        parser.add_argument('--bind_all', action='store_true', help='Serve on all public interfaces. This will expose your TensorBoard instance to\nthe network on both IPv4 and IPv6 (where available). Mutually exclusive with\n`--host`.\n')
        parser.add_argument('--port', metavar='PORT', type=lambda s: None if s == 'default' else int(s), default='default', help='Port to serve TensorBoard on. Pass 0 to request an unused port selected\nby the operating system, or pass "default" to try to bind to the default\nport (%s) but search for a nearby free port if the default port is\nunavailable. (default: "default").' % DEFAULT_PORT)
        parser.add_argument('--reuse_port', metavar='BOOL', type=lambda v: {'true': True, 'false': False}.get(v.lower(), v), choices=[True, False], default=False, help="Enables the SO_REUSEPORT option on the socket opened by TensorBoard's HTTP\nserver, for platforms that support it. This is useful in cases when a parent\nprocess has obtained the port already and wants to delegate access to the\nport to TensorBoard as a subprocess.(default: %(default)s).")
        parser.add_argument('--load_fast', type=str, default='auto', choices=['false', 'auto', 'true'], help='Use alternate mechanism to load data. Typically 100x faster or more, but only\navailable on some platforms and invocations. Defaults to "auto" to use this new\nmode only if available, otherwise falling back to the legacy loading path. Set\nto "true" to suppress the advisory note and hard-fail if the fast codepath is\nnot available. Set to "false" to always fall back. Feedback/issues:\nhttps://github.com/tensorflow/tensorboard/issues/4784\n(default: %(default)s)\n')
        parser.add_argument('--extra_data_server_flags', type=str, default='', help='Experimental. With `--load_fast`, pass these additional command-line flags to\nthe data server. Subject to POSIX word splitting per `shlex.split`. Meant for\ndebugging; not officially supported.\n')
        parser.add_argument('--grpc_creds_type', type=grpc_util.ChannelCredsType, default=grpc_util.ChannelCredsType.LOCAL, choices=grpc_util.ChannelCredsType.choices(), help='Experimental. The type of credentials to use to connect to the data server.\n(default: %(default)s)\n')
        parser.add_argument('--grpc_data_provider', metavar='PORT', type=str, default='', help='Experimental. Address of a gRPC server exposing a data provider. Set to empty\nstring to disable. (default: %(default)s)\n')
        parser.add_argument('--purge_orphaned_data', metavar='BOOL', type=lambda v: {'true': True, 'false': False}.get(v.lower(), v), choices=[True, False], default=True, help='Whether to purge data that may have been orphaned due to TensorBoard\nrestarts. Setting --purge_orphaned_data=False can be used to debug data\ndisappearance. (default: %(default)s)')
        parser.add_argument('--db', metavar='URI', type=str, default='', help='[experimental] sets SQL database URI and enables DB backend mode, which is\nread-only unless --db_import is also passed.')
        parser.add_argument('--db_import', action='store_true', help='[experimental] enables DB read-and-import mode, which in combination with\n--logdir imports event files into a DB backend on the fly. The backing DB is\ntemporary unless --db is also passed to specify a DB path to use.')
        parser.add_argument('--inspect', action='store_true', help='Prints digests of event files to command line.\n\nThis is useful when no data is shown on TensorBoard, or the data shown\nlooks weird.\n\nMust specify one of `logdir` or `event_file` flag.\n\nExample usage:\n  `tensorboard --inspect --logdir mylogdir --tag loss`\n\nSee tensorboard/backend/event_processing/event_file_inspector.py for more info.')
        parser.add_argument('--version_tb', action='store_true', help='Prints the version of Tensorboard')
        parser.add_argument('--tag', metavar='TAG', type=str, default='', help='tag to query for; used with --inspect')
        parser.add_argument('--event_file', metavar='PATH', type=str, default='', help='The particular event file to query for. Only used if --inspect is\npresent and --logdir is not specified.')
        parser.add_argument('--path_prefix', metavar='PATH', type=str, default='', help='An optional, relative prefix to the path, e.g. "/path/to/tensorboard".\nresulting in the new base url being located at\nlocalhost:6006/path/to/tensorboard under default settings. A leading\nslash is required when specifying the path_prefix. A trailing slash is\noptional and has no effect. The path_prefix can be leveraged for path\nbased routing of an ELB when the website base_url is not available e.g.\n"example.site.com/path/to/tensorboard/".')
        parser.add_argument('--window_title', metavar='TEXT', type=str, default='', help='changes title of browser window')
        parser.add_argument('--max_reload_threads', metavar='COUNT', type=int, default=1, help='The max number of threads that TensorBoard can use to reload runs. Not\nrelevant for db read-only mode. Each thread reloads one run at a time.\n(default: %(default)s)')
        parser.add_argument('--reload_interval', metavar='SECONDS', type=_nonnegative_float, default=5.0, help='How often the backend should load more data, in seconds. Set to 0 to\nload just once at startup. Must be non-negative. (default: %(default)s)')
        parser.add_argument('--reload_task', metavar='TYPE', type=str, default='auto', choices=['auto', 'thread', 'process', 'blocking'], help='[experimental] The mechanism to use for the background data reload task.\nThe default "auto" option will conditionally use threads for legacy reloading\nand a child process for DB import reloading. The "process" option is only\nuseful with DB import mode. The "blocking" option will block startup until\nreload finishes, and requires --load_interval=0. (default: %(default)s)')
        parser.add_argument('--reload_multifile', metavar='BOOL', type=lambda v: {'true': True, 'false': False}.get(v.lower(), v), choices=[True, False], default=None, help='[experimental] If true, this enables experimental support for continuously\npolling multiple event files in each run directory for newly appended data\n(rather than only polling the last event file). Event files will only be\npolled as long as their most recently read data is newer than the threshold\ndefined by --reload_multifile_inactive_secs, to limit resource usage. Beware\nof running out of memory if the logdir contains many active event files.\n(default: false)')
        parser.add_argument('--reload_multifile_inactive_secs', metavar='SECONDS', type=int, default=86400, help='[experimental] Configures the age threshold in seconds at which an event file\nthat has no event wall time more recent than that will be considered an\ninactive file and no longer polled (to limit resource usage). If set to -1,\nno maximum age will be enforced, but beware of running out of memory and\nheavier filesystem read traffic. If set to 0, this reverts to the older\nlast-file-only polling strategy (akin to --reload_multifile=false).\n(default: %(default)s - intended to ensure an event file remains active if\nit receives new data at least once per 24 hour period)')
        parser.add_argument('--generic_data', metavar='TYPE', type=str, default='auto', choices=['false', 'auto', 'true'], help='[experimental] Hints whether plugins should read from generic data\nprovider infrastructure. For plugins that support only the legacy\nmultiplexer APIs or only the generic data APIs, this option has no\neffect. The "auto" option enables this only for plugins that are\nconsidered to have stable support for generic data providers. (default:\n%(default)s)')
        parser.add_argument('--samples_per_plugin', type=_parse_samples_per_plugin, default='', help='An optional comma separated list of plugin_name=num_samples pairs to\nexplicitly specify how many samples to keep per tag for that plugin. For\nunspecified plugins, TensorBoard randomly downsamples logged summaries\nto reasonable values to prevent out-of-memory errors for long running\njobs. This flag allows fine control over that downsampling. Note that if a\nplugin is not specified in this list, a plugin-specific default number of\nsamples will be enforced. (for example, 10 for images, 500 for histograms,\nand 1000 for scalars). Most users should not need to set this flag.')
        parser.add_argument('--detect_file_replacement', metavar='BOOL', type=lambda v: {'true': True, 'false': False}.get(v.lower(), v), choices=[True, False], default=None, help='[experimental] If true, this enables experimental support for detecting when\nevent files are replaced with new versions that contain additional data. This is\nnot needed in the normal case where new data is either appended to an existing\nfile or written to a brand new file, but it arises, for example, when using\nrsync without the --inplace option, in which new versions of the original file\nare first written to a temporary file, then swapped into the final location.\n\nThis option is currently incompatible with --load_fast=true, and if passed will\ndisable fast-loading mode. (default: false)')

    def fix_flags(self, flags):
        """Fixes standard TensorBoard CLI flags to parser."""
        FlagsError = base_plugin.FlagsError
        if flags.version_tb:
            pass
        elif flags.inspect:
            if flags.logdir_spec:
                raise FlagsError('--logdir_spec is not supported with --inspect.')
            if flags.logdir and flags.event_file:
                raise FlagsError('Must specify either --logdir or --event_file, but not both.')
            if not (flags.logdir or flags.event_file):
                raise FlagsError('Must specify either --logdir or --event_file.')
        elif flags.logdir and flags.logdir_spec:
            raise FlagsError('May not specify both --logdir and --logdir_spec')
        elif not flags.db and (not flags.logdir) and (not flags.logdir_spec) and (not flags.grpc_data_provider):
            raise FlagsError('A logdir or db must be specified. For example `tensorboard --logdir mylogdir` or `tensorboard --db sqlite:~/.tensorboard.db`. Run `tensorboard --helpfull` for details and examples.')
        elif flags.host is not None and flags.bind_all:
            raise FlagsError('Must not specify both --host and --bind_all.')
        elif flags.load_fast == 'true' and flags.detect_file_replacement is True:
            raise FlagsError('Must not specify both --load_fast=true and--detect_file_replacement=true')
        flags.path_prefix = flags.path_prefix.rstrip('/')
        if flags.path_prefix and (not flags.path_prefix.startswith('/')):
            raise FlagsError('Path prefix must start with slash, but got: %r.' % flags.path_prefix)

    def load(self, context):
        """Creates CorePlugin instance."""
        return CorePlugin(context, include_debug_info=self._include_debug_info)