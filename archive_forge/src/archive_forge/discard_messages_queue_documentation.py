from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
Emulates a Cloud API status queue but drops all messages.

  This is useful when you want to perform some operations but not have the UI
  thread display information about those ops (e.g. running a test or fetching
  the public gsutil tarball object's metadata to perform a version check).
  