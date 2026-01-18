from Snake import Snake
from Constants import NO_OF_CELLS, BANNER_HEIGHT
from Utility import Grid
from DFS import DFS
from BFS import BFS
from A_STAR import A_STAR
from GA import *
def update_GA_ai(self):
    if not self.snake and (not self.model_loaded):
        if self.algo.done():
            if self.algo.next_generation():
                self.snakes = self.algo.population.snakes
            else:
                self.end = True
        for snake in self.snakes:
            x, y = self.algo.run_algorithm(snake)
            snake.move_ai(x, y)
            self.algo.died(snake)
            self.ate_fruit_GA(snake)
    else:
        x, y = self.algo.run_algorithm(self.snake)
        self.snake.move_ai(x, y)
        self.died()
        self.ate_fruit()