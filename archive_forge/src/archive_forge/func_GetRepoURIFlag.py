from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetRepoURIFlag(positional=True, required=True, help_override=None, metavar=None):
    """Get REPO_URI flag."""
    help_txt = help_override or "      Git repository URI containing 1 or more packages as where:\n\n      * REPO_URI - URI of a git repository containing 1 or more packages as\n        subdirectories. In most cases the .git suffix should be specified to\n        delimit the REPO_URI from the PKG_PATH, but this is not required for\n        widely recognized repo prefixes.  If REPO_URI cannot be parsed then\n        an error will be printed an asking for '.git' to be specified\n        as part of the argument. e.g. https://github.com/kubernetes/examples.git\n\n      * PKG_PATH (optional) - Path to Git subdirectory containing Anthos package files.\n       Uses '/' as the path separator (regardless of OS). e.g. staging/cockroachdb.\n       Defaults to the root directory.\n\n      * GIT_REF (optional)- A git tag, branch, ref or commit for the remote version of the\n        package to fetch. Defaults to the repository default branch e.g. @main\n  "
    if not metavar:
        metavar = 'REPO_URI[.git]/[PKG_PATH][@GIT_REF]'
    return GetFlagOrPositional(name='repo_uri', positional=positional, required=required, help=help_txt, metavar=metavar)