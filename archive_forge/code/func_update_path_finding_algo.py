from Snake import Snake
from Constants import NO_OF_CELLS, BANNER_HEIGHT
from Utility import Grid
from DFS import DFS
from BFS import BFS
from A_STAR import A_STAR
from GA import *
def update_path_finding_algo(self, pos):
    if pos == None:
        x, y = self.keep_moving()
    else:
        x = pos.x
        y = pos.y
    self.snake.move_ai(x, y)
    self.died()
    self.ate_fruit()