import sys
from fixtures import Fixture
class PackagePathEntry(Fixture):
    """Add a path to the path of a python package.

    The python package needs to be already imported.

    If this new path is already in the packages __path__ list then the __path__
    list will not be altered.
    """

    def __init__(self, packagename, directory):
        """Create a PackagePathEntry.

        :param directory: The directory to add to the package.__path__.
        """
        self.packagename = packagename
        self.directory = directory

    def _setUp(self):
        path = sys.modules[self.packagename].__path__
        if self.directory in path:
            return
        self.addCleanup(path.remove, self.directory)
        path.append(self.directory)