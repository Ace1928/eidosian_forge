from pyparsing import *
import random
import string
class TakeCommand(Command):

    def __init__(self, quals):
        super(TakeCommand, self).__init__('TAKE', 'taking')
        self.subject = quals.item

    @staticmethod
    def helpDescription():
        return 'TAKE or PICKUP or PICK UP - pick up an object (but some are deadly)'

    def _doCommand(self, player):
        rm = player.room
        subj = Item.items[self.subject]
        if subj in rm.inv and subj.isVisible:
            if subj.isTakeable:
                rm.removeItem(subj)
                player.take(subj)
            else:
                print(subj.cantTakeMessage)
        else:
            print('There is no %s here.' % subj)