from pyparsing import *
import random
import string
class Room(object):

    def __init__(self, desc):
        self.desc = desc
        self.inv = []
        self.gameOver = False
        self.doors = [None, None, None, None]

    def __getattr__(self, attr):
        return {'n': self.doors[0], 's': self.doors[1], 'e': self.doors[2], 'w': self.doors[3]}[attr]

    def enter(self, player):
        if self.gameOver:
            player.gameOver = True

    def addItem(self, it):
        self.inv.append(it)

    def removeItem(self, it):
        self.inv.remove(it)

    def describe(self):
        print(self.desc)
        visibleItems = [it for it in self.inv if it.isVisible]
        if random.random() > 0.5:
            if len(visibleItems) > 1:
                is_form = 'are'
            else:
                is_form = 'is'
            print('There {0} {1} here.'.format(is_form, enumerateItems(visibleItems)))
        else:
            print('You see %s.' % enumerateItems(visibleItems))