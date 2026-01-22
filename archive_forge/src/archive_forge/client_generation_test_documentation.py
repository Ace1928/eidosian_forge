import importlib
import logging
import os
import six
import subprocess
import sys
import tempfile
import unittest
from apitools.gen import gen_client
from apitools.gen import test_utils
Test gen_client against all the APIs we use regularly.