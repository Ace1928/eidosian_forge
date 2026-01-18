from pyparsing import *
import random
import string
def useItem(self, player, target):
    if self.useAction:
        self.useAction(player, self, target)