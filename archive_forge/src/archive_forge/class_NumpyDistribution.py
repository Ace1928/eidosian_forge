from distutils.core import Distribution
class NumpyDistribution(Distribution):

    def __init__(self, attrs=None):
        self.scons_data = []
        self.installed_libraries = []
        self.installed_pkg_config = {}
        Distribution.__init__(self, attrs)

    def has_scons_scripts(self):
        return bool(self.scons_data)