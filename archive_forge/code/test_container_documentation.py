import ddt
import time
from zunclient.tests.functional.osc.v1 import base
Check container execute command with name and UUID arguments.

        Test steps:
        1) Create container in setUp.
        2) Execute command calling it with name and UUID arguments.
        3) Check the container logs.
        