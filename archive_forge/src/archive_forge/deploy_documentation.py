import argparse
import logging
import traceback
from pathlib import Path
from textwrap import dedent
from deploy_lib import (MAJOR_VERSION, Config, cleanup, config_option_exists,
 pyside6-deploy deployment tool

    Deployment tool that uses Nuitka to deploy PySide6 applications to various desktop (Windows,
    Linux, macOS) platforms.

    How does it work?

    Command: pyside6-deploy path/to/main_file
             pyside6-deploy (incase main file is called main.py)
             pyside6-deploy -c /path/to/config_file

    Platforms supported: Linux, Windows, macOS
    Module binary inclusion:
        1. for non-QML cases, only required modules are included
        2. for QML cases, all modules are included because of all QML plugins getting included
            with nuitka

    Config file:
        On the first run of the tool, it creates a config file called pysidedeploy.spec which
        controls the various characteristic of the deployment. Users can simply change the value
        in this config file to achieve different properties ie. change the application name,
        deployment platform etc.

        Note: This file is used by both pyside6-deploy and pyside6-android-deploy

