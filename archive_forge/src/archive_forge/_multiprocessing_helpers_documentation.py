import os
import warnings
Helper module to factorize the conditional multiprocessing import logic

We use a distinct module to simplify import statements and avoid introducing
circular dependencies (for instance for the assert_spawning name).
