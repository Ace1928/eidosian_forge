import base64
import sys
import mock
from pyu2f import errors
from pyu2f import model
from pyu2f.convenience import localauthenticator
Test signing with multiple keys registered and one is eligible.