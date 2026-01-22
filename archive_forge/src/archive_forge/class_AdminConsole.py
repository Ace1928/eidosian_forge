from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import re
import string
import sys
import wsgiref.util
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import backendinfo
from googlecloudsdk.third_party.appengine._internal import six_subset
class AdminConsole(validation.Validated):
    """Class representing an admin console directives in application info."""
    ATTRIBUTES = {PAGES: validation.Optional(validation.Repeated(AdminConsolePage))}

    @classmethod
    def Merge(cls, adminconsole_one, adminconsole_two):
        """Returns the result of merging two `AdminConsole` objects."""
        if not adminconsole_one or not adminconsole_two:
            return adminconsole_one or adminconsole_two
        if adminconsole_one.pages:
            if adminconsole_two.pages:
                adminconsole_one.pages.extend(adminconsole_two.pages)
        else:
            adminconsole_one.pages = adminconsole_two.pages
        return adminconsole_one