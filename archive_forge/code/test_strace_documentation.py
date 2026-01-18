import threading
from breezy import strace, tests
from breezy.strace import StraceResult, strace_detailed
from breezy.tests.features import strace_feature
Checks that a reasonable raw strace log was found by strace.