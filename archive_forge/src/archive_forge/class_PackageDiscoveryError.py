from distutils import errors as _distutils_errors
class PackageDiscoveryError(BaseError, RuntimeError):
    """Impossible to perform automatic discovery of packages and/or modules.

    The current project layout or given discovery options can lead to problems when
    scanning the project directory.

    Setuptools might also refuse to complete auto-discovery if an error prone condition
    is detected (e.g. when a project is organised as a flat-layout but contains
    multiple directories that can be taken as top-level packages inside a single
    distribution [*]_). In these situations the users are encouraged to be explicit
    about which packages to include or to make the discovery parameters more specific.

    .. [*] Since multi-package distributions are uncommon it is very likely that the
       developers did not intend for all the directories to be packaged, and are just
       leaving auxiliary code in the repository top-level, such as maintenance-related
       scripts.
    """