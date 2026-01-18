import sys
import os
@property
def user_data_dir(self):
    return user_data_dir(self.appname, self.appauthor, version=self.version, roaming=self.roaming)