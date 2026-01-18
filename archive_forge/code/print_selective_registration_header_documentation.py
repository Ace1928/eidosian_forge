import argparse
import sys
from absl import app
from tensorflow.python.tools import selective_registration_header_lib
Prints a header file to be used with SELECTIVE_REGISTRATION.

An example of command-line usage is:
  bazel build tensorflow/python/tools:print_selective_registration_header && \
  bazel-bin/tensorflow/python/tools/print_selective_registration_header \
    --graphs=path/to/graph.pb > ops_to_register.h

Then when compiling tensorflow, include ops_to_register.h in the include search
path and pass -DSELECTIVE_REGISTRATION and -DSUPPORT_SELECTIVE_REGISTRATION
 - see core/framework/selective_registration.h for more details.

When compiling for Android:
  bazel build -c opt --copt="-DSELECTIVE_REGISTRATION" \
    --copt="-DSUPPORT_SELECTIVE_REGISTRATION" \
    //tensorflow/tools/android/inference_interface:libtensorflow_inference.so \
    --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
    --crosstool_top=//external:android/crosstool --cpu=armeabi-v7a
