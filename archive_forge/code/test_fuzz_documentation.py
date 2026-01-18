import os
import pytest
from bs4 import (
This file contains test cases reported by third parties using
fuzzing tools, primarily from Google's oss-fuzz project. Some of these
represent real problems with Beautiful Soup, but many are problems in
libraries that Beautiful Soup depends on, and many of the test cases
represent different ways of triggering the same problem.

Grouping these test cases together makes it easy to see which test
cases represent the same problem, and puts the test cases in close
proximity to code that can trigger the problems.
