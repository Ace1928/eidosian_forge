from pyparsing import *
import random
import string
def useShovel(p, subj, target):
    coin = Item.items['coin']
    if not coin.isVisible and coin in p.room.inv:
        coin.isVisible = True