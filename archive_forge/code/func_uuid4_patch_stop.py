import os
import uuid
import xmltodict
from pytest import skip, fixture
from mock import patch
def uuid4_patch_stop():
    uuid4_patcher.stop()