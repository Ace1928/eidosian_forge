import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
class RepoEntry:

    def __init__(self, url, rev, timestamp):
        self.url = url
        self.rev = rev
        self.timestamp = timestamp

    def __str__(self):
        return 'repo: %s;%s  %s' % (self.url, self.rev, self.timestamp)