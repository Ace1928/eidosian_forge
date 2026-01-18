import pygame
from Constants import *
from GA import *
import sys
def train_GA(self):
    self.game.curr_menu = self.game.main_menu
    self.run_display = False
    self.game.curr_menu.state = 'GA'
    self.game.playing = True
    if len(self.no_population.input) > 0:
        Population.population = int(self.no_population.input)
    if len(self.no_hidden_nodes.input) > 0:
        Population.hidden_node = int(self.no_hidden_nodes.input)
    if len(self.no_generation.input) > 0:
        GA.generation = int(self.no_generation.input)
    if len(self.mutation_rate.input) > 0:
        GA.mutation_rate = int(self.mutation_rate.input) / 100