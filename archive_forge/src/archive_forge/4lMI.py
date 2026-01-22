from Algorithm import Algorithm
from Snake import Snake
import math
import random
from Utility import Node
from Constants import NO_OF_CELLS, BANNER_HEIGHT, USER_SEED
import numpy as np
import logging

# Setting up logging configuration with maximum verbosity and detail
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Seed the random number generator for reproducibility, ensuring deterministic behavior
random.seed(USER_SEED)


class Population:
    population_size: int = 300
    hidden_node_count: int = 8

    def __init__(self) -> None:
        self.snakes: list[Snake] = []
        self.saved_snakes: list[Snake] = []

    def initialize_population(self) -> None:
        logging.debug("Initializing population of snakes with meticulous detail.")
        for _ in range(Population.population_size):
            new_snake = Snake(Population.hidden_node_count)
            self.snakes.append(new_snake)
            logging.debug(
                f"Added new snake with ID {id(new_snake)} to population, enhancing the genetic diversity."
            )

    def remove_snake(self, snake: Snake) -> None:
        logging.debug(
            f"Removing snake with ID {id(snake)} from active population, transitioning to saved snakes."
        )
        self.saved_snakes.append(snake)
        self.snakes.remove(snake)
        logging.debug(
            f"Snake with ID {id(snake)} has been successfully moved to saved snakes, preserving its genetic data for future generations."
        )


class GA(Algorithm):
    max_generations: int = 30
    mutation_rate: float = 0.12

    def __init__(self, grid: np.ndarray) -> None:
        super().__init__(grid)
        self.population = Population()
        self.current_generation: int = 0
        self.best_score: int = 0
        self.best_generation: int = 0
        self.best_snake: Snake = None

    def process_snake_death(self, snake: Snake) -> None:
        logging.debug(
            f"Processing potential death for snake with ID {id(snake)}, evaluating conditions."
        )
        current_x = snake.body[0].x
        current_y = snake.body[0].y

        if snake.ate_body() or snake.life_time > 80:
            logging.info(
                f"Snake with ID {id(snake)} died due to self-collision or exceeding life time threshold."
            )
            self.population.remove_snake(snake)

        elif (
            not 0 <= current_x < NO_OF_CELLS
            or not BANNER_HEIGHT <= current_y < NO_OF_CELLS
        ):
            logging.info(
                f"Snake with ID {id(snake)} died due to boundary collision, initiating removal."
            )
            self.population.remove_snake(snake)

    def generate_next_generation(self) -> bool:
        if self.current_generation == GA.max_generations:
            logging.info(
                "Maximum generation limit reached, halting further generation production."
            )
            return False

        self.calculate_fitness()
        self.identify_best_snake()
        self.perform_natural_selection()
        self.population.saved_snakes.clear()
        self.current_generation += 1
        logging.debug(
            f"Advanced to generation {self.current_generation}, preparing for new evolutionary challenges."
        )
        return True

    def is_population_extinct(self) -> bool:
        return len(self.population.snakes) == 0

    def identify_best_snake(self) -> Snake:
        best_snake = max(self.population.saved_snakes, key=lambda snake: snake.fitness)
        logging.debug(
            f"Identified best snake with ID {id(best_snake)} and fitness {best_snake.fitness}, potentially setting a new benchmark for performance."
        )

        if best_snake.score > self.best_score:
            self.best_score = best_snake.score
            self.best_generation = self.current_generation
            self.best_snake = best_snake
            logging.info(
                f"New best snake found with score {self.best_score} at generation {self.best_generation}, marking a significant milestone in the evolutionary process."
            )

        return best_snake

    def check_directions(
        self, snake: Snake, direction_node: Node, inputs: list
    ) -> None:
        if self.outside_boundary(direction_node) or self.inside_body(
            snake, direction_node
        ):
            inputs.append(1)
        else:
            inputs.append(0)

    def run_algorithm(self, snake: Snake) -> tuple:
        inputs: list = []
        fruit_node: Node = Node(snake.get_fruit().x, snake.get_fruit().y)

        # head direction
        x: int = snake.body[0].x
        y: int = snake.body[0].y

        if snake.body[1].x == x:
            # left = Node(x-1, y)
            # right = Node(x+1, y)

            if snake.body[1].y < y:
                # going down
                forward: Node = Node(x, y + 1)
                left: Node = Node(x - 1, y)
                right: Node = Node(x + 1, y)
            else:
                # going up
                forward: Node = Node(x, y - 1)
                left: Node = Node(x + 1, y)
                right: Node = Node(x - 1, y)

        elif snake.body[1].y == y:
            # left = Node(x, y+1)
            # right = Node(x, y-1)

            if snake.body[1].x < x:
                # going right
                forward: Node = Node(x + 1, y)
                left: Node = Node(x, y - 1)
                right: Node = Node(x, y + 1)
            else:
                # going left
                forward: Node = Node(x - 1, y)
                left: Node = Node(x, y + 1)
                right: Node = Node(x, y - 1)

        # Check potential directions
        self.check_directions(snake, forward, inputs)
        self.check_directions(snake, left, inputs)
        self.check_directions(snake, right, inputs)

        # Calculate distances to the fruit
        forward_distance: float = self.euclidean_distance(fruit_node, forward)
        left_distance: float = self.euclidean_distance(fruit_node, left)
        right_distance: float = self.euclidean_distance(fruit_node, right)

        distances: list[float] = [forward_distance, left_distance, right_distance]
        min_index: int = distances.index(min(distances))

        # Append the direction with the minimum distance to the inputs
        inputs.append(min_index)

        # Calculate angle between the head and the fruit
        head_vector: np.ndarray = np.array([int(snake.body[0].x), int(snake.body[0].y)])
        fruit_vector: np.ndarray = np.array([fruit_node.x, fruit_node.y])

        inner_product: float = np.inner(head_vector, fruit_vector)
        norms_product: float = np.linalg.norm(head_vector) * np.linalg.norm(
            fruit_vector
        )

        cosine_angle: float = round(inner_product / norms_product, 5)
        sine_angle: float = math.sqrt(1 - cosine_angle**2)
        inputs.append(sine_angle)

        # Feed inputs through the neural network
        outputs: list = snake.network.feedforward(inputs)

        # Determine the best direction based on network output
        max_index: int = outputs.index(max(outputs))
        chosen_direction: Node = {0: forward, 1: left, 2: right}[max_index]

        logging.debug(
            f"Snake with ID {id(snake)} will move to {chosen_direction.x}, {chosen_direction.y}, following the optimal path as determined by neural computation."
        )
        return chosen_direction.x, chosen_direction.y

    def select_parent(self) -> Snake:
        index: int = 0
        r: float = random.random()

        while r > 0:
            r -= self.population.saved_snakes[index].fitness
            index += 1
        index -= 1

        selected_parent: Snake = self.population.saved_snakes[index]
        logging.debug(
            f"Selected parent snake with ID {id(selected_parent)} for reproduction, ensuring genetic diversity and robustness in the next generation."
        )
        return selected_parent

    def perform_natural_selection(self) -> None:
        new_snakes: list[Snake] = []
        for _ in range(Population.population_size):
            parent_a: Snake = self.select_parent()
            parent_b: Snake = self.select_parent()
            child: Snake = Snake(Population.hidden_node_count)
            child.network.crossover(parent_a.network, parent_b.network)
            child.network.mutate(GA.mutation_rate)

            new_snakes.append(child)
            logging.debug(
                f"Created new snake with ID {id(child)} from parents {id(parent_a)} and {id(parent_b)}, contributing to the evolutionary progress through genetic crossover and mutation."
            )

        self.population.snakes = new_snakes.copy()
        logging.info(
            "Completed natural selection and created new generation of snakes, ensuring the continuation and enhancement of the species through meticulous genetic management."
        )

    def calculate_fitness(self) -> None:
        for snake in self.population.saved_snakes:
            fitness: float = (snake.steps**3) * (3 ** (snake.score * 3)) - 1.5 ** (
                0.25 * snake.steps
            )
            snake.fitness = round(fitness, 7)
            logging.debug(
                f"Calculated fitness for snake with ID {id(snake)}: {snake.fitness}, quantifying its performance and potential for survival in a competitive environment."
            )
        self.normalize_fitness_values()

    def normalize_fitness_values(self) -> None:
        total_fitness: float = sum(
            snake.fitness for snake in self.population.saved_snakes
        )
        for snake in self.population.saved_snakes:
            snake.fitness /= total_fitness
            logging.debug(
                f"Normalized fitness for snake with ID {id(snake)} to {snake.fitness}, ensuring equitable comparison and selection based on relative performance."
            )

    def done(self) -> bool:
        """
        Determine if the genetic algorithm process is complete.
        """
        logging.debug(
            f"Checking if GA has completed: Current generation {self.current_generation} / Max generations {GA.max_generations}"
        )
        return self.current_generation >= GA.max_generations
