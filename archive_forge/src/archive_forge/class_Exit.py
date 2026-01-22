from pyparsing import *
import random
import string
class Exit(Room):

    def __init__(self):
        super(Exit, self).__init__('')

    def enter(self, player):
        player.gameOver = True