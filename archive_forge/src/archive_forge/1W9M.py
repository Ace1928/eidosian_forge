import pygame
import sys
import logging
from Constants import *
from GA import *

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Menu:
    def __init__(self, game: 'Game') -> None:
        logging.debug("Initializing Menu class.")
        self.game: 'Game' = game
        self.mid_size: float = self.game.SIZE / 2
        self.run_display: bool = True
        self.cursor_rect: pygame.Rect = pygame.Rect(0, 0, 20, 20)
        self.offset: int = -150
        self.title_size: int = 50
        self.option_size: int = 28
        logging.debug("Menu class initialized with mid_size: {}, run_display: {}, cursor_rect: {}, offset: {}, title_size: {}, option_size: {}.".format(self.mid_size, self.run_display, self.cursor_rect, self.offset, self.title_size, self.option_size))

    def draw_cursor(self) -> None:
        logging.debug("Drawing cursor on menu.")
        try:
            self.game.draw_text(
                '*', size=20,
                x=self.cursor_rect.x, y=self.cursor_rect.y,
                color=MENU_COLOR
            )
            logging.debug("Cursor drawn at position: ({}, {}).".format(self.cursor_rect.x, self.cursor_rect.y))
        except Exception as e:
            logging.error("Failed to draw cursor: {}".format(e), exc_info=True)
            raise RuntimeError("Drawing cursor failed.") from e

    def blit_menu(self) -> None:
        logging.debug("Blitting menu to the screen.")
        try:
            self.game.window.blit(self.game.display, (0, 0))
            pygame.display.update()
            self.game.reset_keys()
            logging.debug("Menu blitted successfully.")
        except Exception as e:
            logging.error("Failed to blit menu: {}".format(e), exc_info=True)
            raise RuntimeError("Blitting menu failed.") from e

class MainMenu(Menu):
    def __init__(self, game: 'Game') -> None:
        logging.debug("Initializing MainMenu class.")
        super().__init__(game)
        self.state: str = 'BFS'
        self.cursorBFS: tuple = MENU_COLOR
        self.cursorDFS: tuple = WHITE
        self.cursorASTAR: tuple = WHITE
        self.cursorGA: tuple = WHITE
        self.BFSx, self.BFSy = self.mid_size, self.mid_size - 50
        self.DFSx, self.DFSy = self.mid_size, self.mid_size + 0
        self.ASTARx, self.ASTARy = self.mid_size, self.mid_size + 50
        self.GAx, self.GAy = self.mid_size, self.mid_size + 100
        self.cursor_rect.midtop = (self.BFSx + self.offset, self.BFSy)
        logging.debug("MainMenu class initialized with state: {}, cursor colors set, and cursor positions defined.")

    def change_cursor_color(self) -> None:
        logging.debug("Changing cursor color based on state: {}.".format(self.state))
        try:
            self.clear_cursor_color()
            if self.state == 'BFS':
                self.cursorBFS = MENU_COLOR
            elif self.state == 'DFS':
                self.cursorDFS = MENU_COLOR
            elif self.state == 'ASTAR':
                self.cursorASTAR = MENU_COLOR
            elif self.state == 'GA':
                self.cursorGA = MENU_COLOR
            logging.debug("Cursor color changed to reflect current state.")
        except Exception as e:
            logging.error("Failed to change cursor color: {}".format(e), exc_info=True)
            raise RuntimeError("Changing cursor color failed.") from e

    def clear_cursor_color(self) -> None:
        logging.debug("Clearing cursor colors.")
        self.cursorBFS = WHITE
        self.cursorDFS = WHITE
        self.cursorASTAR = WHITE
        self.cursorGA = WHITE
        logging.debug("Cursor colors cleared.")

    def display_menu(self) -> None:
        logging.debug("Displaying main menu.")
        self.run_display = True
        try:
            while self.run_display:
                self.game.event_handler()
                self.check_input()
                self.game.display.fill(WINDOW_COLOR)
                self.game.draw_text(
                    'Ai Snake Game', size=self.title_size,
                    x=self.game.SIZE/2, y=self.game.SIZE/2 - 2*(CELL_SIZE + NO_OF_CELLS),
                    color=TITLE_COLOR
                )
                self.game.draw_text(
                    'BFS', size=self.option_size,
                    x=self.BFSx,  y=self.BFSy,
                    color=self.cursorBFS
                )
                self.game.draw_text(
                    'DFS', size=self.option_size,
                    x=self.DFSx,  y=self.DFSy,
                    color=self.cursorDFS
                )
                self.game.draw_text(
                    'AStar', size=self.option_size,
                    x=self.ASTARx,  y=self.ASTARy,
                    color=self.cursorASTAR
                )
                self.game.draw_text(
                    'Genetic Algorithm', size=self.option_size,
                    x=self.GAx,  y=self.GAy,
                    color=self.cursorGA
                )
                self.draw_cursor()
                self.change_cursor_color()
                self.blit_menu()
            logging.debug("Main menu displayed and interactive.")
        except Exception as e:
            logging.error("Failed to display main menu: {}".format(e), exc_info=True)
            raise RuntimeError("Displaying main menu failed.") from e

    def check_input(self) -> None:
        logging.debug("Checking user input for menu navigation.")
        try:
            self.move_cursor()
            if self.game.START:
                if self.state == 'GA':  # go to genetic algorithm options
                    self.game.curr_menu = self.game.GA
                else:
                    self.game.playing = True
                self.run_display = False
                logging.debug("Input processed and state updated.")
        while self.run_display:
            self.game.event_handler()
            self.check_input()

            self.game.display.fill(WINDOW_COLOR)

            self.game.draw_text(
                'Ai Snake Game', size=self.title_size,
                x=self.game.SIZE/2, y=self.game.SIZE/2 - 2*(CELL_SIZE + NO_OF_CELLS),
                color=TITLE_COLOR
            )

            self.game.draw_text(
                'BFS', size=self.option_size,
                x=self.BFSx,  y=self.BFSy,
                color=self.cursorBFS
            )
            self.game.draw_text(
                'DFS', size=self.option_size,
                x=self.DFSx,  y=self.DFSy,
                color=self.cursorDFS
            )

            self.game.draw_text(
                'AStar', size=self.option_size,
                x=self.ASTARx,  y=self.ASTARy,
                color=self.cursorASTAR
            )

            self.game.draw_text(
                'Genetic Algorithm', size=self.option_size,
                x=self.GAx,  y=self.GAy,
                color=self.cursorGA
            )

            self.draw_cursor()
            self.change_cursor_color()
            self.blit_menu()

    def check_input(self):
        self.move_cursor()

        if self.game.START:
            if self.state == 'GA':  # go to genetic algorith options
                self.game.curr_menu = self.game.GA
            else:
                self.game.playing = True
            self.run_display = False

    def move_cursor(self):
        if self.game.DOWNKEY:
            if self.state == 'BFS':
                self.cursor_rect.midtop = (
                    self.DFSx + self.offset, self.DFSy)
                self.state = 'DFS'

            elif self.state == 'DFS':
                self.cursor_rect.midtop = (
                    self.ASTARx + self.offset, self.ASTARy)
                self.state = 'ASTAR'

            elif self.state == 'ASTAR':
                self.cursor_rect.midtop = (
                    self.GAx + self.offset, self.GAy)
                self.state = 'GA'

            elif self.state == 'GA':
                self.cursor_rect.midtop = (
                    self.BFSx + self.offset, self.BFSy)
                self.state = 'BFS'

        if self.game.UPKEY:
            if self.state == 'BFS':
                self.cursor_rect.midtop = (
                    self.GAx + self.offset, self.GAy)
                self.state = 'GA'

            elif self.state == 'DFS':
                self.cursor_rect.midtop = (
                    self.BFSx + self.offset, self.BFSy)
                self.state = 'BFS'

            elif self.state == 'ASTAR':
                self.cursor_rect.midtop = (
                    self.DFSx + self.offset, self.DFSy)
                self.state = 'DFS'

            elif self.state == 'GA':
                self.cursor_rect.midtop = (
                    self.ASTARx + self.offset, self.ASTARy)
                self.state = 'ASTAR'


class button():
    def __init__(self, x, y, text, game):
        self.x = x
        self.y = y
        self.text = text
        self.game = game
        self.font = pygame.font.Font(game.font_name, 30)
        self.clicked = False

    def draw_button(self):
        action = False

        # get mouse position
        pos = pygame.mouse.get_pos()

        # create pygame Rect object for the button
        button_rect = pygame.Rect(self.x, self.y, BTN_WIDTH, BTN_HEIGHT)

        # check mouseover and clicked conditions
        if button_rect.collidepoint(pos):
            if pygame.mouse.get_pressed()[0] == 1:
                self.clicked = True
                pygame.draw.rect(self.game.display, BTN_CLICKED, button_rect)
            elif pygame.mouse.get_pressed()[0] == 0 and self.clicked == True:
                self.clicked = False
                action = True
            else:
                pygame.draw.rect(self.game.display, BTN_HOVER, button_rect)
        else:
            pygame.draw.rect(self.game.display, BTN_COLOR, button_rect)

        # add text to button
        text_img = self.font.render(self.text, True, WHITE)
        text_len = text_img.get_width()
        self.game.display.blit(text_img, (self.x + int(BTN_WIDTH / 2) -
                                          int(text_len / 2), self.y + 25))

        return action


class TextBox:
    def __init__(self, x, y, game):
        self.font = pygame.font.Font(game.font_name, 20)
        self.input_rect = pygame.Rect(x, y, TXT_WIDTH, TXT_HEIGHT)
        self.input = ''
        self.game = game
        self.active = False

    def draw_input(self):
        # get mouse position
        pos = pygame.mouse.get_pos()

        if self.input_rect.collidepoint(pos):
            if pygame.mouse.get_pressed()[0] == 1:
                self.active = True

        elif pygame.mouse.get_pressed()[0] == 1:
            self.active = False

        if self.active:
            color = TXT_ACTIVE
        else:
            color = TXT_PASSIVE

        pygame.draw.rect(self.game.display, color, self.input_rect, 2)
        text_surface = self.font.render(self.input, False, WHITE)
        self.game.display.blit(
            text_surface, (self.input_rect.x + 15, self.input_rect.y + 1))


class GAMenu(Menu):
    def __init__(self, game, controller):
        Menu.__init__(self, game)

        self.controller = controller
        self.train_model = button(
            game.SIZE/2 - 4*(CELL_SIZE + NO_OF_CELLS), game.SIZE/2 + 3.5*(CELL_SIZE + NO_OF_CELLS), 'Train Model', game)
        self.load_model = button(
            game.SIZE/2 + (CELL_SIZE), game.SIZE/2 + 3.5*(CELL_SIZE + NO_OF_CELLS), 'Load Model', game)

        self.no_population = TextBox(
            self.game.SIZE/2 + 50, self.game.SIZE/2 - 60, game)
        self.no_generation = TextBox(
            self.game.SIZE/2 + 50, self.game.SIZE/2 - 10, game)
        self.no_hidden_nodes = TextBox(
            self.game.SIZE/2 + 50, self.game.SIZE/2 + 40, game)
        self.mutation_rate = TextBox(
            self.game.SIZE/2 + 50, self.game.SIZE/2 + 90, game)
        self.init_input()

    def init_input(self):
        self.no_population.input = '300'
        self.no_generation.input = '30'
        self.no_hidden_nodes.input = '8'
        self.mutation_rate.input = '12'

    def display_menu(self):
        self.run_display = True
        while self.run_display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game.running, self.game.playing = False, False
                    self.game.curr_menu.run_display = False
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.game.BACK = True

                    if self.no_population.active:
                        if event.key == pygame.K_BACKSPACE:
                            self.no_population.input = self.no_population.input[:-1]
                        else:
                            if event.unicode.isdigit() and len(self.no_population.input) < 3:
                                self.no_population.input += event.unicode

                    if self.no_generation.active:
                        if event.key == pygame.K_BACKSPACE:
                            self.no_generation.input = self.no_generation.input[:-1]
                        else:
                            if event.unicode.isdigit() and len(self.no_generation.input) < 3:
                                self.no_generation.input += event.unicode

                    if self.no_hidden_nodes.active:
                        if event.key == pygame.K_BACKSPACE:
                            self.no_hidden_nodes.input = self.no_hidden_nodes.input[:-1]
                        else:
                            if event.unicode.isdigit() and len(self.no_hidden_nodes.input) < 2:
                                self.no_hidden_nodes.input += event.unicode

                    if self.mutation_rate.active:
                        if event.key == pygame.K_BACKSPACE:
                            self.mutation_rate.input = self.mutation_rate.input[:-1]
                        else:
                            if event.unicode.isdigit() and len(self.mutation_rate.input) < 3:
                                self.mutation_rate.input += event.unicode

            self.check_input()
            self.game.display.fill(WINDOW_COLOR)

            self.game.draw_text(
                'GA Options', self.title_size,
                self.game.SIZE/2, self.game.SIZE /
                2 - 4*(CELL_SIZE + NO_OF_CELLS),
                color=TITLE_COLOR
            )
            self.game.draw_text(
                'Settings to train model:', 25,
                self.game.SIZE/2,
                self.game.SIZE/2 - 2*(CELL_SIZE + NO_OF_CELLS),
                color=MENU_COLOR
            )
            self.game.draw_text(
                "No. of populations     : ", 20,
                self.game.SIZE/2 - 2*CELL_SIZE,
                self.game.SIZE/2 - 50,
                color=BANNER_COLOR
            )

            self.game.draw_text(
                "No. of generations     : ", 20,
                self.game.SIZE/2 - 2*CELL_SIZE,
                self.game.SIZE/2,
                color=BANNER_COLOR
            )

            self.game.draw_text(
                "No. of hidden nodes   : ", 20,
                self.game.SIZE/2 - 2*CELL_SIZE,
                self.game.SIZE/2 + 50,
                color=BANNER_COLOR
            )

            self.game.draw_text(
                "Mutation rate %:          : ", 20,
                self.game.SIZE/2 - 2*CELL_SIZE,
                self.game.SIZE/2 + 100,
                color=BANNER_COLOR
            )

            self.no_population.draw_input()
            self.no_generation.draw_input()
            self.no_hidden_nodes.draw_input()

            self.mutation_rate.draw_input()

            if self.load_model.draw_button():
                self.load_GA()
            if self.train_model.draw_button():
                self.train_GA()

            self.game.draw_text(
                "Q to return to main menu", 20,
                self.game.SIZE/2, self.game.SIZE/2 + 6*(NO_OF_CELLS+CELL_SIZE),
                color=WHITE
            )

            self.blit_menu()
        self.reset()

    def reset(self):
        self.no_population.active = False
        self.no_generation.active = False
        self.no_hidden_nodes.active = False
        self.mutation_rate.active = False

    def check_input(self):
        if self.game.BACK:
            self.game.curr_menu = self.game.main_menu
            self.run_display = False

    def load_GA(self):
        self.game.curr_menu = self.game.main_menu
        self.run_display = False
        self.game.curr_menu.state = 'GA'
        self.game.playing = True
        self.game.load_model = True

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
            GA.mutation_rate = int(self.mutation_rate.input)/100
