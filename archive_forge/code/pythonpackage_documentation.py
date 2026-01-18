import os.path
from fixtures import Fixture
from fixtures._fixtures.tempdir import TempDir
Create a PythonPackage.

        :param packagename: The name of the package to create - e.g.
            'toplevel.subpackage.'
        :param modulelist: List of modules to include in the package.
            Each module should be a tuple with the filename and content it
            should have.
        :param init: If false, do not create a missing __init__.py. When
            True, if modulelist does not include an __init__.py, an empty
            one is created.
        