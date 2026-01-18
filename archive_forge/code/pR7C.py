from typing import Optional, Tuple, List
import logging
from Snake import Snake
from Constants import NO_OF_CELLS, BANNER_HEIGHT
from Utility import Grid, Node
from DFS import DFS
from BFS import BFS
from A_STAR import A_STAR
from GA import GA, Population
from pygame.math import Vector2

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class GameController:

    def __init__(self) -> None:
        logging.debug("Initializing GameController")
        self.snake: Optional[Snake] = None
        self.snakes: List[Snake] = []
        self.score: int = 0
        self.end: bool = False
        self.grid: List[List[Node]] = Grid().grid
        self.algo: Optional[GA | BFS | DFS | A_STAR] = None
        self.model_loaded: bool = False
        logging.debug("GameController initialized with empty parameters")

    def reset(self) -> None:
        logging.debug("Resetting GameController state")
        self.end = False
        if self.snake:
            self.snake.reset()
            self.snake = None

        self.algo = None
        self.snakes = []
        self.model_loaded = False
        logging.debug("GameController state reset complete")

    def best_GA_score(self) -> int:
        logging.debug("Fetching best GA score")
        return self.algo.best_score if self.algo and isinstance(self.algo, GA) else 0

    def best_GA_gen(self) -> int:
        logging.debug("Fetching best GA generation")
        return self.algo.best_gen if self.algo and isinstance(self.algo, GA) else 0

    def curr_gen(self) -> int:
        logging.debug("Fetching current GA generation")
        return self.algo.generation if self.algo and isinstance(self.algo, GA) else 0

    def save_model(self) -> None:
        logging.debug("Saving model")
        if isinstance(self.algo, GA):
            best_snake = self.algo.best_snake
            if best_snake:
                network = best_snake.network
                best_snake.save_model(network, "saved_model")
                logging.debug("Model saved successfully")
            else:
                logging.error("No best snake to save model from")
        else:
            logging.error("Algorithm is not GA, cannot save model")

    def load_model(self) -> None:
        logging.debug("Loading model")
        self.snake = Snake()
        self.snake.load_model("saved_model")
        self.model_loaded = True
        logging.debug("Model loaded successfully")

    def get_score(self) -> int:
        logging.debug("Fetching score")
        return self.snake.score if self.snake else 0

    def ate_fruit(self) -> None:
        logging.debug("Checking if snake ate fruit")
        if self.snake and self.snake.ate_fruit():
            self.snake.add_body_ai()
            self.change_fruit_location()
            logging.debug("Snake ate fruit and body was added")

    def change_fruit_location(self) -> None:
        logging.debug("Changing fruit location")
        while True:
            self.snake.create_fruit()
            position: Vector2 = self.snake.get_fruit()
            inside_body: bool = any(position == body for body in self.snake.body)

            if not inside_body:
                logging.debug(f"Fruit placed at {position}")
                break
            else:
                logging.debug(f"Collision detected at {position}, retrying")

    def ate_fruit_GA(self, snake: Snake) -> None:
        logging.debug(f"Checking if GA snake {snake} ate fruit")
        if snake.ate_fruit():
            snake.add_body_ai()
            self.change_fruit_location_GA(snake)
            logging.debug("GA snake ate fruit and body was added")

    def change_fruit_location_GA(self, snake: Snake) -> None:
        logging.debug(f"Changing fruit location for GA snake {snake}")
        while True:
            snake.create_fruit()
            position: Vector2 = snake.get_fruit()
            inside_body: bool = any(position == body for body in snake.body)

            if not inside_body:
                logging.debug(f"Fruit placed at {position} for GA snake {snake}")
                break
            else:
                logging.debug(
                    f"Collision detected at {position} for GA snake {snake}, retrying"
                )

    def died(self) -> None:
        logging.debug("Checking if snake has died")
        current_x: int = self.snake.body[0].x
        current_y: int = self.snake.body[0].y

        if not 0 <= current_x < NO_OF_CELLS:
            self.end = True
            logging.debug("Snake died by moving out of horizontal bounds")
        elif not BANNER_HEIGHT <= current_y < NO_OF_CELLS:
            self.end = True
            logging.debug("Snake died by moving out of vertical bounds")
        elif self.snake.ate_body():
            self.end = True
            logging.debug("Snake died by eating its own body")

    def get_fruit_pos(self) -> Vector2:
        logging.debug("Fetching fruit position")
        return self.snake.get_fruit()

    def set_algorithm(self, algo_type: str) -> None:
        logging.debug(f"Setting algorithm type to {algo_type}")
        if self.algo is not None:
            logging.debug("Algorithm already set, skipping")
            return

        if algo_type == "BFS":
            self.algo = BFS(self.grid)
            self.snake = Snake()
            logging.debug("BFS algorithm set")

        elif algo_type == "DFS":
            self.algo = DFS(self.grid)
            self.snake = Snake()
            logging.debug("DFS algorithm set")

        elif algo_type == "ASTAR":
            self.algo = A_STAR(self.grid)
            self.snake = Snake()
            logging.debug("A* algorithm set")

        elif algo_type == "GA":
            self.algo = GA(self.grid)
            if not self.model_loaded:
                self.algo.population._initialpopulation_()
                self.snakes = self.algo.population.snakes
                logging.debug("GA algorithm set and initial population created")

    def ai_play(self, algorithm: str) -> None:
        logging.debug(f"AI play initiated with algorithm {algorithm}")
        self.set_algorithm(algorithm)

        if self.algo is None:
            logging.error("Algorithm not set, aborting AI play")
            return

        if isinstance(self.algo, GA):
            self.update_GA_ai()
        else:
            pos: Optional[Node] = self.algo.run_algorithm(self.snake)
            self.update_path_finding_algo(pos)

    def keep_moving(self) -> Tuple[int, int]:
        logging.debug("Continuing movement based on last direction")
        x: int = self.snake.body[0].x
        y: int = self.snake.body[0].y

        if self.snake.body[1].x == x:
            if self.snake.body[1].y < y:
                # keep going down
                y = y + 1
            else:
                # keep going up
                y = y - 1
        elif self.snake.body[1].y == y:
            if self.snake.body[1].x < x:
                # keep going right
                x = x + 1
            else:
                # keep going left
                x = x - 1
        logging.debug(f"New position determined: ({x}, {y})")
        return x, y

    def update_GA_ai(self) -> None:
        logging.debug("Updating GA AI")
        if not self.snake and not self.model_loaded:
            if self.algo.done():
                if self.algo.next_generation():
                    self.snakes = self.algo.population.snakes
                    logging.debug("Moved to next generation in GA")
                else:
                    self.end = True
                    logging.debug("GA algorithm ended, no more generations")

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

    def update_path_finding_algo(self, pos: Optional[Node]) -> None:
        logging.debug("Updating path-finding algorithm")
        if pos is None:
            x, y = self.keep_moving()
        else:
            x = pos.x
            y = pos.y

        self.snake.move_ai(x, y)
        self.died()
        self.ate_fruit()
