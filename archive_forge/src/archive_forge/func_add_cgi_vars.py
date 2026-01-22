from .util import FileWrapper, guess_scheme, is_hop_by_hop
from .headers import Headers
import sys, os, time
def add_cgi_vars(self):
    self.environ.update(self.base_env)