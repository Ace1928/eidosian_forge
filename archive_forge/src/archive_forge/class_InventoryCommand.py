from pyparsing import *
import random
import string
class InventoryCommand(Command):

    def __init__(self, quals):
        super(InventoryCommand, self).__init__('INV', 'taking inventory')

    @staticmethod
    def helpDescription():
        return 'INVENTORY or INV or I - lists what items you have'

    def _doCommand(self, player):
        print('You have %s.' % enumerateItems(player.inv))