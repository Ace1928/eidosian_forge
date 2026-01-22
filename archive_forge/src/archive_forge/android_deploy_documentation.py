import argparse
import logging
import shutil
import traceback
from pathlib import Path
from textwrap import dedent
from deploy_lib import (setup_python, create_config_file, cleanup, install_python_dependencies,
from deploy_lib.android import AndroidData, AndroidConfig
from deploy_lib.android.buildozer import Buildozer
 pyside6-android-deploy deployment tool

    Deployment tool that uses buildozer (https://buildozer.readthedocs.io/en/latest/) and
    python-for-android (https://python-for-android.readthedocs.io/en/latest/) to deploy PySide6
    applications to Android

    How does it work?

    Command: pyside6-android-deploy --wheel-pyside=<pyside_wheel_path>
                                    --wheel-shiboken=<shiboken_wheel_path>
                                    --ndk-path=<optional_ndk_path>
                                    --sdk-path=<optional_sdk_path>
             pyside6-android-deploy android -c /path/to/pysidedeploy.spec


    Note: If --ndk-path and --sdk-path are not specified, the cache of the tool
    `.pyside6_android_deploy` is checked in the user's HOME directory. If it is not found, the user
    will have to manually download them.

    Prerequisities: Python main entrypoint file should be named "main.py"

    Platforms Supported: aarch64, armv7a, i686, x86_64

    Config file:
        On the first run of the tool, it creates a config file called pysidedeploy.spec which
        controls the various characteristic of the deployment. Users can simply change the value
        in this config file to achieve different properties ie. change the application name,
        deployment platform etc.

        Note: This file is used by both pyside6-deploy and pyside6-android-deploy
