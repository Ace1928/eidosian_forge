import sys
import os
@property
def user_cache_dir(self):
    return user_cache_dir(self.appname, self.appauthor, version=self.version)