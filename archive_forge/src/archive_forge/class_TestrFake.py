from distutils import cmd
import distutils.errors
import logging
import os
import sys
import warnings
class TestrFake(cmd.Command):
    description = 'Run unit tests using testr'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print("Install testrepository to run 'testr' command properly.")