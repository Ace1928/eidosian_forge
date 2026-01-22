import gyp.input
import unittest
def _create_dependency(self, dependent, dependency):
    dependent.dependencies.append(dependency)
    dependency.dependents.append(dependent)