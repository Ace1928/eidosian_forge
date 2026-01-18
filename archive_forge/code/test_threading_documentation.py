import pytest
import rpy2.rinterface
import rpy2.rinterface_lib.embedded
from threading import Thread
from rpy2.rinterface import embedded
Wrapper around Thread allowing to record exceptions from the thread.