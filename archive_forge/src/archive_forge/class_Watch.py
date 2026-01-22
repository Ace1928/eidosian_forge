import logging
import re
class Watch(object):
    """Watch line entry.

    This will contain the attributes documented in uscan(1):

    :ivar url: The URL (possibly including the filename regex)
    :ivar matching_pattern: a filename regex, optional
    :ivar version: version policy, optional
    :ivar script: script to run, optional
    :ivar opts: a list of options, as strings
    """

    def __init__(self, url, matching_pattern=None, version=None, script=None, opts=None):
        self.url = url
        self.matching_pattern = matching_pattern
        self.version = version
        self.script = script
        if opts is None:
            opts = []
        self.options = opts

    def __repr__(self):
        return '%s(%r, matching_pattern=%r, version=%r, script=%r, opts=%r)' % (self.__class__.__name__, self.url, self.matching_pattern, self.version, self.script, self.options)

    def __eq__(self, other):
        if not isinstance(other, Watch):
            return False
        return other.url == self.url and other.matching_pattern == self.matching_pattern and (other.version == self.version) and (other.script == self.script) and (other.options == self.options)