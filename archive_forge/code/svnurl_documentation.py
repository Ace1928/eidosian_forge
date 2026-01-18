import os, sys, time, re
import py
from py import path, process
from py._path import common
from py._path import svnwc as svncommon
from py._path.cacheutil import BuildcostAccessCache, AgingCache
 return a list of LogEntry instances for this path.
rev_start is the starting revision (defaulting to the first one).
rev_end is the last revision (defaulting to HEAD).
if verbose is True, then the LogEntry instances also know which files changed.
