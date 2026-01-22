from pyparsing import *
import random
import string
class MoveCommand(Command):

    def __init__(self, quals):
        super(MoveCommand, self).__init__('MOVE', 'moving')
        self.direction = quals.direction[0]

    @staticmethod
    def helpDescription():
        return "MOVE or GO - go NORTH, SOUTH, EAST, or WEST\n          (can abbreviate as 'GO N' and 'GO W', or even just 'E' and 'S')"

    def _doCommand(self, player):
        rm = player.room
        nextRoom = rm.doors[{'N': 0, 'S': 1, 'E': 2, 'W': 3}[self.direction]]
        if nextRoom:
            player.moveTo(nextRoom)
        else:
            print("Can't go that way.")