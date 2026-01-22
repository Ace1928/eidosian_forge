from pyparsing import *
import random
import string
class QuitCommand(Command):

    def __init__(self, quals):
        super(QuitCommand, self).__init__('QUIT', 'quitting')

    @staticmethod
    def helpDescription():
        return 'QUIT or Q - ends the game'

    def _doCommand(self, player):
        print('Ok....')
        player.gameOver = True