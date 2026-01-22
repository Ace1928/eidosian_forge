from pyparsing import *
import random
import string
class LookCommand(Command):

    def __init__(self, quals):
        super(LookCommand, self).__init__('LOOK', 'looking')

    @staticmethod
    def helpDescription():
        return 'LOOK or L - describes the current room and any objects in it'

    def _doCommand(self, player):
        player.room.describe()