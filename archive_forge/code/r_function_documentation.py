from .. import utils
from .._lazyload import rpy2
from . import conversion
import functools
Install a Github repository.

    Parameters
    ----------
    repo: string
        Github repository name to install.
    lib: string
        Directory to install the package.
        If missing, defaults to the first element of .libPaths().
    dependencies: boolean, optional (default: None/NA)
        When True, installs all packages specified under "Depends", "Imports",
        "LinkingTo" and "Suggests".
        When False, installs no dependencies.
        When None/NA, installs all packages specified under "Depends", "Imports"
        and "LinkingTo".
    update: string or boolean, optional (default: False)
        One of "default", "ask", "always", or "never". "default"
        Respects R_REMOTES_UPGRADE variable if set, falls back to "ask" if unset.
        "ask" prompts the user for which out of date packages to upgrade.
        For non-interactive sessions "ask" is equivalent to "always".
        TRUE and FALSE also accepted, correspond to "always" and "never" respectively.
    type : {"binary", "source", "both"}, optional (default: "binary")
        Which package version to install if a newer version is available as source.
        "both" tries source first and uses binary as a fallback.
    build_vignettes: boolean, optional (default: False)
        Builds Github vignettes.
    force: boolean, optional (default: False)
        Forces installation even if remote state has not changed since previous install.
    verbose: boolean, optional (default: True)
        Install script verbosity.
    