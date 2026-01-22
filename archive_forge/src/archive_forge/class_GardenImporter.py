from the legacy structure `garden.flower` to the newer `flower` structure used
import importlib.util
from importlib.abc import MetaPathFinder
import sys
from os.path import dirname, join, realpath, exists, abspath
from kivy import kivy_home_dir
from kivy.utils import platform
import kivy
class GardenImporter(MetaPathFinder):

    def find_spec(self, fullname, path, target=None):
        if path != 'kivy.garden':
            return None
        moddir = join(garden_kivy_dir, fullname.split('.', 2)[-1], '__init__.py')
        if exists(moddir):
            return importlib.util.spec_from_file_location(fullname, moddir)
        modname = fullname.split('.', 1)[-1]
        for directory in (garden_app_dir, garden_system_dir):
            moddir = join(directory, modname, '__init__.py')
            if exists(moddir):
                return importlib.util.spec_from_file_location(fullname, moddir)
        return None