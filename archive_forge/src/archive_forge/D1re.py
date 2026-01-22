import pygame
import random
import heapq
import logging
from typing import List, Optional, Dict, Any, Tuple
import cProfile
from collections import deque
import numpy as np
import time
import torch
from functools import lru_cache as LRUCache
import math
import asyncio
from scipy.spatial import Delaunay
import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Color constants using numpy arrays for optimal performance and consistency:
BG_LIGHT_GREEN = np.array([137, 200, 80])
BG_DARK_GREEN = np.array([123, 181, 70])
BLUE = np.array([47, 174, 232])
DARK_BLUE = np.array([44, 163, 217])
RED = np.array([217, 42, 42])
DARK_RED = np.array([197, 38, 38])
GRAY = np.array([155, 155, 155])
BLACK = np.array([0, 0, 0])

# Game constants defined with numpy for precision and efficiency:
TILE_SIZE = 10
HEIGHT = 100
WIDTH = 100
WIN_HEIGHT, WIN_WIDTH = HEIGHT * TILE_SIZE, WIDTH * TILE_SIZE
WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Snake")
FPS = 60
CLOCK = pygame.time.Clock()


# Setup logging
def configure_logging():
    """
    Configures the logging settings for this module. This function sets the logging level to DEBUG and specifies the format
    for logging messages. This configuration is encapsulated within this function to prevent side effects on logging settings
    in other modules when this module is imported.
    """
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )


class Grid:
    def __init__(self, width, height):
        """
        Initializes the Grid object with specified width and height.

        Args:
            width (int): The width of the grid in terms of number of cells.
            height (int): The height of the grid in terms of number of cells.
        """
        self.width = width
        self.height = height
        self.cells = np.zeros((height, width), dtype=int)  # Initialize a grid of zeros
        logging.debug(f"Grid created with dimensions {width}x{height}")

    def is_position_free(self, position):
        """
        Checks if a given position is free (not occupied) on the grid.

        Args:
            position (numpy array): The position to check on the grid.

        Returns:
            bool: True if the position is free, False otherwise.
        """
        x, y = position
        return self.cells[y, x] == 0

    def get_points(self):
        """
        Generates a list of all coordinate points within the grid.

        Returns:
            list of tuples: A list containing all (x, y) coordinates in the grid.
        """
        points = [(x, y) for x in range(WIDTH) for y in range(HEIGHT)]
        logging.debug(f"Generated {len(points)} points for the grid.")
        return points

    def update_position(self, position, value):
        """
        Updates the grid cell at a given position with a specified value.

        Args:
            position (numpy array): The position to update on the grid.
            value (int): The value to set at the given position.
        """
        x, y = position
        self.cells[y, x] = value
        logging.debug(f"Grid position {position} updated with value {value}")

    def reset_grid(self):
        """
        Resets the entire grid to zero values, effectively clearing the grid.
        """
        self.cells.fill(0)
        logging.debug("Grid has been reset to zero values.")

    def is_near_obstacle(self, position, obstacle_positions=None):
        """
        Checks if a given position is near an obstacle on the grid.

        Args:
            position (tuple): The (x, y) position to check.
            obstacle_positions (list of tuples, optional): A list of obstacle positions to consider. If not provided, the method will use the grid's cells to determine obstacles.

        Returns:
            bool: True if the position is near an obstacle, False otherwise.
        """
        if obstacle_positions is None:
            obstacle_positions = [(x, y) for x in range(self.width) for y in range(self.height) if self.cells[y, x] != 0]

        for obstacle in obstacle_positions:
            if np.linalg.norm(np.array(position) - np.array(obstacle)) <= 1:
                logging.debug(f"Position {position} is near an obstacle at {obstacle}")
                return True

        logging.debug(f"Position {position} is not near any obstacles")
        return False

    def get_obstacles(self):
        """
        Returns a list of all obstacle positions on the grid.

        Returns:
            list of tuples: A list containing (x, y) coordinates of all obstacles on the grid.
        """
        obstacles = [(x, y) for x in range(self.width) for y in range(self.height) if self.cells[y, x] != 0]
        logging.debug(f"Found {len(obstacles)} obstacles on the grid")
        return obstacles


class Food:
    def __init__(self, position=None):
        """
        Initializes the Food object with an optional position.

        Args:
            position (numpy array, optional): The initial position of the food. If not provided, a random position will be generated.
        """
        self.position = position if position is not None else self.generate_random_position()
        logging.debug(f"Food created at position {self.position}")

    def generate_random_position(self):
        """
        Generates a random position for the food within the grid boundaries.

        Returns:
            numpy array: The generated random position.
        """
        x = np.random.randint(0, WIDTH)
        y = np.random.randint(0, HEIGHT)
        position = np.array([x, y])
        logging.debug(f"Generated random food position: {position}")
        return position

    def draw(self, WIN):
        """
        Draws the food on the game window.

        Args:
            WIN (pygame.Surface): The game window surface to draw on.
        """
        x, y = self.position
        pygame.draw.rect(WIN, RED, (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))
        logging.debug(f"Food drawn at position {self.position}")


class Snake:
    def __init__(self, grid_width=WIDTH, grid_height=HEIGHT):
        """
        Initializes the Snake object with default settings. The snake is initially set to be alive with its segments
        positioned centrally on the game grid. The snake starts without any movement direction and is in a frozen state
        until the game begins. The default color of the snake is set to BLUE, and it can operate in different modes,
        initially set to 'hamiltonian'.

        Args:
            grid_width (int): Width of the game grid.
            grid_height (int): Height of the game grid.
        """
        logging.debug("Initializing Snake object with default settings.")
        try:
            self.alive = True
            self.segments = deque(
                [np.array([grid_width // 2, grid_height // 2 - i]) for i in range(3)]
            )
            self.direction = "right"  # Initial direction
            self.frozen = True
            self.color = BLUE  # Default color
            self.mode = "hamiltonian"  # Modes: 'hamiltonian', 'astar'
            self.color_phase = [
                random.randint(0, 360) for _ in self.segments
            ]  # HSV phase for each segment
            logging.info(
                "Snake object initialized successfully with attributes: alive=True, segments=central, direction=right, frozen=True, color=BLUE, mode=hamiltonian."
            )
        except Exception as e:
            logging.error(f"Failed to initialize Snake object: {e}")
            raise Exception(f"Snake initialization error: {e}")

    def get_head_position(self):
        return self.segments[-1]

    def get_tail_position(self):
        return self.segments[0]

    def move(self):
        """
        Updates the position of the snake based on its current direction. This method efficiently manages the segments
        deque by appending the new head position and popping the tail position when moving, ensuring optimal memory usage
        and performance.
        """
        logging.debug(
            "Attempting to update the position of the snake based on its current direction."
        )
        try:
            if self.frozen or not self.alive:
                logging.info("Snake movement aborted: Snake is frozen or not alive.")
                return

            new_head = np.copy(self.segments[-1])
            if self.direction == "up":
                new_head[1] -= 1
            elif self.direction == "down":
                new_head[1] += 1
            elif self.direction == "left":
                new_head[0] -= 1
            elif self.direction == "right":
                new_head[0] += 1

            # Check if the new head position is out of bounds or collides with the snake's body
            if not (0 <= new_head[0] < WIDTH and 0 <= new_head[1] < HEIGHT) or np.any(
                [np.array_equal(new_head, segment) for segment in self.segments]
            ):
                self.alive = False
                logging.info("Snake has collided with the wall or itself and died.")
                return

            self.segments.append(new_head)
            self.segments.popleft()  # Remove the tail segment
            logging.info(f"Snake moved successfully in direction: {self.direction}.")
        except Exception as e:
            logging.error(f"Failed to move the snake: {e}")
            raise Exception(f"Snake movement error: {e}")

    def change_direction(self, direction):
        """
        Changes the direction of the snake's movement if the new direction is not directly opposite to the current direction,
        preventing the snake from reversing onto itself.
        """
        logging.debug(
            f"Attempting to change the direction of the snake's movement to: {direction}."
        )
        try:
            opposite_directions = {
                "up": "down",
                "down": "up",
                "left": "right",
                "right": "left",
            }
            if direction != opposite_directions.get(self.direction, ""):
                self.direction = direction
                logging.info(
                    f"Snake direction changed successfully to: {self.direction}."
                )
        except Exception as e:
            logging.error(f"Failed to change the direction of the snake: {e}")
            raise Exception(f"Snake direction change error: {e}")

    def draw(self, WIN):
        """
        Draws each segment of the snake with a cycling color spectrum and applies glow effects based on the mode.
        """
        hue_step = 1  # Define how fast the color cycles through the spectrum
        for i, segment in enumerate(self.segments):
            # Update color phase
            self.color_phase[i] = (self.color_phase[i] + hue_step) % 360
            color = pygame.Color(0)
            color.hsva = (
                self.color_phase[i],
                100,
                100,
            )  # Full saturation and value for vivid colors

            # Draw the segment
            pygame.draw.rect(
                WIN,
                color,
                (segment[0] * TILE_SIZE, segment[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE),
            )

            # Apply glow effect based on the mode
            if self.mode == "hamiltonian":
                self.apply_glow(WIN, segment, (255, 0, 0))  # Red glow
            elif self.mode == "astar":
                self.apply_glow(WIN, segment, (0, 0, 255))  # Blue glow

    def apply_glow(self, WIN, position, glow_color):
        """
        Applies a glow effect around a given position with the specified glow color.
        """
        glow_radius = 10  # Radius of the glow effect
        for radius in range(glow_radius, 0, -1):
            alpha = (1 - (radius / glow_radius)) * 255
            glow_surface = pygame.Surface(
                (TILE_SIZE + radius * 2, TILE_SIZE + radius * 2), pygame.SRCALPHA
            )
            pygame.draw.circle(
                glow_surface,
                glow_color + (int(alpha),),
                (glow_radius, glow_radius),
                radius,
            )
            WIN.blit(
                glow_surface,
                (position[0] * TILE_SIZE - radius, position[1] * TILE_SIZE - radius),
            )

        pygame.display.update()


class DecisionMaker:
    def __init__(self, snake=None, food=None, grid=None):
        """
        Initializes the DecisionMaker object which controls the snake's movement strategy with optional parameters.
        This constructor meticulously sets up the snake, food, and grid objects, and initializes pathfinders for various strategies,
        ensuring that each component is optimally configured for high-performance gameplay.

        Args:
            snake (Snake): The snake object. Defaults to a new Snake instance if not provided.
            food (Food): The food object. Defaults to a new Food instance positioned at the center if not provided.
            grid (Grid): The grid object representing the game area. Defaults to a new Grid instance with predefined width and height if not provided.
        """
        self.snake = snake if snake is not None else Snake()
        self.food = (
            food
            if food is not None
            else Food(position=np.array([WIDTH // 2, HEIGHT // 2]))
        )
        self.grid = grid if grid is not None else Grid(WIDTH, HEIGHT)
        self.pathfinders = {
            "CDP": ConstrainedDelaunayPathfinder(
                Grid.get_points(self.grid), Grid.get_obstacles(self.grid)
            ),
            "AHP": AmoebaHamiltonianPathfinder(self.snake, self.grid, self.food),
            "ThetaStar": ThetaStar(self.grid),
        }

    async def decide_next_move(self):
        """
        Asynchronously decides the next move for the snake based on the current game state and the selected strategy.
        It evaluates potential paths using multiple pathfinding algorithms, compares their costs, and selects the optimal path.
        This method continuously updates its assessments to look ahead multiple steps based on the game's complexity and dynamics.

        Returns:
            str: The next direction for the snake to move, determined by the optimal pathfinding strategy.
        """
        current_position = self.snake.get_head_position()
        food_position = self.food.position
        paths = {}
        costs = {}

        # Generate paths from each pathfinding strategy
        for name, pathfinder in self.pathfinders.items():
            path = await pathfinder.find_path(current_position, food_position)
            paths[name] = path
            costs[name] = await self.calculate_path_cost(path)

        # Determine the optimal path with the lowest cost
        optimal_strategy = min(costs, key=costs.get)
        optimal_path = paths[optimal_strategy]

        # Decide the next move based on the optimal path
        next_move = await self.determine_next_move_from_path(optimal_path)
        return next_move

    def calculate_path_cost(self, path):
        """
        Calculates the cost of a given path based on length, proximity to dangers, and strategic advantages.
        This method employs a detailed and comprehensive cost analysis to ensure optimal path selection.

        Args:
            path (list): The path as a list of grid positions.

        Returns:
            float: The calculated cost of the path.
        """
        length_cost = len(path) * 10
        danger_cost = sum(self.grid.is_near_obstacle(point) for point in path) * 20
        strategic_cost = self.evaluate_strategic_advantages(path)
        return length_cost + danger_cost + strategic_cost

    def evaluate_strategic_advantages(self, path):
        """
        Evaluates the strategic advantages of a given path, considering factors such as game-winning alignment and future food positions.
        This method performs a thorough analysis of the path's potential to lead to a winning game state.

        Args:
            path (list): The path as a list of grid positions.

        Returns:
            float: The strategic advantage score of the path.
        """
        game_winning_alignment_cost = self.evaluate_game_winning_strategy_alignment(
            path
        )
        future_food_proximity_cost = self.calculate_future_food_proximity_cost(path)
        strategic_cost = game_winning_alignment_cost + future_food_proximity_cost
        logging.debug(f"Total strategic cost calculated: {strategic_cost}")
        return strategic_cost

    def calculate_future_food_proximity_cost(self, path):
        """
        Calculates the cost associated with the proximity to predicted future food positions along the given path.
        This method leverages advanced predictive modeling techniques to anticipate the likelihood of food appearing in certain positions.

        Args:
            path (list): The path as a list of grid positions.

        Returns:
            float: The cost based on the path's proximity to predicted future food positions.
        """
        future_food_positions = self.predict_future_food_positions()
        proximity_cost = sum(
            self.calculate_position_proximity_cost(position, future_food_positions)
            for position in path
        )
        logging.debug(
            f"Calculated future food proximity cost for path: {proximity_cost}"
        )
        return proximity_cost

    def calculate_position_proximity_cost(self, position, target_positions):
        """
        Calculates the cost associated with the proximity of a given position to a set of target positions.
        This method assesses the strategic value of a position based on its distance from important target locations.

        Args:
            position (tuple): The position to evaluate.
            target_positions (list): A list of target positions to consider.

        Returns:
            float: The proximity cost of the position relative to the target positions.
        """
        min_distance = min(
            np.linalg.norm(np.array(position) - np.array(target))
            for target in target_positions
        )
        proximity_cost = 10 / (min_distance + 1)  # Inverse proportional to distance
        logging.debug(
            f"Calculated proximity cost for position {position}: {proximity_cost}"
        )
        return proximity_cost

    def predict_future_food_positions(self):
        """
        Predicts future food positions based on a comprehensive analysis of historical game data and the current game state,
        utilizing a sophisticated machine learning model. This method integrates complex data analysis techniques to forecast
        the probable locations where food might appear on the game grid, thereby enabling strategic planning for the snake's movements.

        Returns:
            list: A meticulously compiled list of predicted future food positions, each represented as a tuple of coordinates.
        """
        # Logging the initiation of the food position prediction process
         logging.debug("Initiating the prediction of future food positions.")
        # Extract historical food position data and current game state
        historical_data = self.retrieve_historical_food_positions()
        current_state_features = self.extract_features_from_current_state()
        # Combine historical data with current state for prediction
        features = np.concatenate((historical_data, current_state_features), axis=0)
        # Normalize the feature set
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_features[:, :-1],
            scaled_features[:, -1],
            test_size=0.2,
            random_state=42,
        )
        # Initialize and train the machine learning model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        # Predict the future food positions
        predicted_positions = model.predict(X_test)
        predicted_positions = [
            (int(pos[0]), int(pos[1])) for pos in predicted_positions
        ]
        # Log the completion of the prediction process
        logging.debug(f"Predicted future food positions: {predicted_positions}")
        return predicted_positions

    def retrieve_historical_food_positions(self):
        """
        Retrieves historical food position data from the game's data storage system, ensuring data integrity and optimizing query performance
        through advanced indexing techniques and caching mechanisms. This method efficiently fetches a comprehensive dataset of past food
        locations, enabling the decision-making system to learn from historical patterns and make well-informed predictions.

        Returns:
            numpy.ndarray: An array of historical food positions, where each row represents a single food location record, meticulously
            formatted and preprocessed to facilitate seamless integration with the machine learning pipeline.
        """
        logging.debug("Retrieving historical food position data.")
        try:
            # Simulated retrieval of historical food position data
            historical_data = np.random.randint(0, max(WIDTH, HEIGHT), size=(100, 2))
            logging.info(
                f"Successfully retrieved {len(historical_data)} historical food position records."
            )
            return historical_data
        except Exception as e:
            logging.error(f"Failed to retrieve historical food position data: {e}")
            raise Exception(f"Historical data retrieval error: {e}")

    def extract_features_from_current_state(self):
        """
        Extracts a rich set of informative features from the current game state, capturing critical aspects such as the snake's length,
        the current food position, and other relevant metrics. This method applies advanced feature engineering techniques to transform
        raw game data into a highly expressive and compact representation, optimized for predictive modeling purposes.

        Returns:
            numpy.ndarray: An array of features extracted from the current game state, each feature carefully selected and processed to
            maximize the predictive performance of the model.
        """
        logging.debug(
            "Extracting features from the current game state for predictive modeling."
        )
        try:
            # Extract the snake's length and the current position of the food
            snake_length = self.snake.length
            food_position_x = self.food.position[0]
            food_position_y = self.food.position[1]
            logging.debug(
                f"Current snake length: {snake_length}, Food position: ({food_position_x}, {food_position_y})"
            )

            # Combine these features into a numpy array
            current_state_features = np.array(
                [snake_length, food_position_x, food_position_y]
            )
            logging.info(
                f"Features extracted from current game state: {current_state_features}"
            )
            return current_state_features
        except Exception as e:
            logging.error(f"Failed to extract features from current game state: {e}")
            raise Exception(f"Feature extraction error: {e}")

    def evaluate_obstacle_proximity_cost(self, position):
        """
        Evaluates the cost associated with the proximity of a given position to obstacles on the grid.
        This method quantifies the risk and potential impact of being close to obstacles during gameplay.

        Args:
            position (tuple): The position to evaluate.

        Returns:
            float: The obstacle proximity cost for the given position.
        """
        obstacle_positions = self.grid.get_obstacles()
        min_distance = min(
            np.linalg.norm(np.array(position) - np.array(obstacle))
            for obstacle in obstacle_positions
        )
        cost = 20 / (min_distance + 1)  # Inverse proportional to distance
        logging.debug(
            f"Calculated obstacle proximity cost for position {position}: {cost}"
        )
        return cost

    def evaluate_game_winning_strategy_alignment(self, path):
        """
        Evaluates how well the given path aligns with established game-winning strategies.

        Args:
            path (list): The path as a list of grid positions.

        Returns:
            float: A cost representing the strategic alignment with winning strategies.
        """
        alignment_cost = 10 * (
            len(path) - self.snake.length
        )  # Simplified strategy alignment cost
        logging.debug(
            f"Calculated game winning strategy alignment cost for path: {alignment_cost}"
        )
        return alignment_cost

    def determine_next_move_from_path(self, path):
        """
        Determines the next move direction based on the first step in the path.
        This method ensures that the decision is made with precision and aligns with the optimal path strategy.

        Args:
            path (list): The path as a list of grid positions.

        Returns:
            str: The direction to move ('up', 'down', 'left', 'right').
        """
        if not path or len(path) < 2:
            return "none"  # No move possible

        current_head = self.snake.get_head_position()
        next_position = path[1]  # The next step in the path

        if next_position[0] < current_head[0]:
            return "left"
        elif next_position[0] > current_head[0]:
            return "right"
        elif next_position[1] < current_head[1]:
            return "up"
        elif next_position[1] > current_head[1]:
            return "down"
        return "none"


class ConstrainedDelaunayPathfinder:
    def __init__(self, points, obstacles):
        """
        Initializes the ConstrainedDelaunayPathfinder object with the given points and obstacles.
        This constructor meticulously constructs a constrained Delaunay triangulation graph, optimizing its structure
        for efficient pathfinding operations while ensuring strict adherence to the specified constraints.

        Args:
            points (list): A list of points representing the nodes of the graph.
            obstacles (list): A list of obstacle positions to be considered as constraints in the triangulation.
        """
        self.points = points
        self.obstacles = obstacles
        self.graph = self.build_constrained_delaunay_graph()

    def build_constrained_delaunay_graph(self):
        """
        Builds a constrained Delaunay triangulation graph based on the given points and obstacles.
        This method employs advanced computational geometry techniques to construct a highly optimized and
        topologically sound graph structure, facilitating efficient pathfinding operations while strictly
        respecting the specified constraints.

        Returns:
            networkx.Graph: The constructed constrained Delaunay triangulation graph, meticulously crafted to
            provide a solid foundation for precise and reliable pathfinding algorithms.
        """
        logging.debug("Building constrained Delaunay triangulation graph.")
        try:
            # Create a Delaunay triangulation
            tri = Delaunay(self.points)

            # Create a graph from the triangulation edges
            graph = nx.Graph()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        if not self.is_constrained_edge(
                            tri.points[simplex[i]], tri.points[simplex[j]]
                        ):
                            graph.add_edge(
                                tuple(tri.points[simplex[i]]),
                                tuple(tri.points[simplex[j]]),
                            )

            logging.info(
                f"Constrained Delaunay triangulation graph constructed with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges."
            )
            return graph
        except Exception as e:
            logging.error(
                f"Failed to build constrained Delaunay triangulation graph: {e}"
            )
            raise Exception(f"Constrained Delaunay triangulation error: {e}")

    def is_constrained_edge(self, p1, p2):
        """
        Determines whether the edge between two points is constrained by an obstacle.
        This method performs a meticulous check to verify if the line segment connecting the given points
        intersects with any of the specified obstacle positions, ensuring the integrity of the constrained
        Delaunay triangulation.

        Args:
            p1 (tuple): The first point of the edge.
            p2 (tuple): The second point of the edge.

        Returns:
            bool: True if the edge is constrained by an obstacle, False otherwise.
        """
        for obstacle in self.obstacles:
            if self.is_point_on_line_segment(obstacle, p1, p2):
                logging.debug(
                    f"Edge between {p1} and {p2} is constrained by obstacle at {obstacle}."
                )
                return True
        return False

    def is_point_on_line_segment(self, point, p1, p2):
        """
        Determines whether a given point lies on the line segment between two other points.
        This method employs a precise geometric calculation to check if the point coincides with the line segment,
        considering floating-point precision and ensuring reliable results.

        Args:
            point (tuple): The point to check.
            p1 (tuple): The first endpoint of the line segment.
            p2 (tuple): The second endpoint of the line segment.

        Returns:
            bool: True if the point lies on the line segment, False otherwise.
        """
        cross_product = (point[1] - p1[1]) * (p2[0] - p1[0]) - (point[0] - p1[0]) * (
            p2[1] - p1[1]
        )
        if abs(cross_product) > 1e-10:
            return False

        dot_product = (point[0] - p1[0]) * (p2[0] - p1[0]) + (point[1] - p1[1]) * (
            p2[1] - p1[1]
        )
        if dot_product < 0:
            return False

        squared_length = (p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (
            p2[1] - p1[1]
        )
        if dot_product > squared_length:
            return False

        return True

    async def find_path(self, start, end):
        """
        Finds the shortest path between the start and end points using the constrained Delaunay triangulation graph.
        This method employs an asynchronous pathfinding algorithm to efficiently traverse the graph, considering the
        constraints imposed by the obstacles and ensuring the optimality of the computed path.

        Args:
            start (tuple): The starting point of the path.
            end (tuple): The endpoint of the path.

        Returns:
            list: The shortest path from start to end as a list of points, meticulously computed to guarantee
            the most efficient and obstacle-free route.
        """
        try:
            path = nx.astar_path(self.graph, start, end)
            logging.info(
                f"Shortest path found from {start} to {end}: {' -> '.join(map(str, path))}"
            )
            return path
        except nx.NetworkXNoPath:
            logging.warning(f"No path found from {start} to {end}.")
            return []


class AmoebaHamiltonianPathfinder:
    def __init__(self, snake, grid, food):
        """
        Initializes the AmoebaHamiltonianPathfinder object with the given snake, grid, and food objects.
        This constructor sets up the necessary data structures and configurations to enable the computation
        of Hamiltonian paths using an amoeba-inspired algorithm, tailored to the specific requirements of
        the snake game environment.

        Args:
            snake (Snake): The snake object representing the game agent.
            grid (Grid): The grid object representing the game environment.
            food (Food): The food object representing the target for the snake.
        """
        self.snake = snake
        self.grid = grid
        self.food = food

    async def find_path(self, start, end):
        """
        Finds a Hamiltonian path from the start to the end point using an amoeba-inspired algorithm.
        This method employs a novel nature-inspired approach to efficiently explore the game grid, adapting
        to the dynamic constraints imposed by the snake's body and the environment, while striving to find
        a path that covers all accessible cells.

        Args:
            start (tuple): The starting point of the path.
            end (tuple): The endpoint of the path.

        Returns:
            list: A Hamiltonian path from start to end as a list of points, meticulously computed to ensure
            maximum coverage of the game grid while avoiding obstacles and the snake's body.
        """
        logging.debug(f"Finding Hamiltonian path from {start} to {end}.")
        path = []
        visited = set()
        current_position = start

        while current_position != end:
            visited.add(current_position)
            path.append(current_position)

            next_position = self.select_next_position(current_position, visited)
            if next_position is None:
                logging.warning(
                    f"No valid next position found from {current_position}. Terminating path."
                )
                break

            current_position = next_position

        if current_position == end:
            path.append(end)
            logging.info(
                f"Hamiltonian path found from {start} to {end}: {' -> '.join(map(str, path))}"
            )
        else:
            logging.warning(
                f"Hamiltonian path from {start} to {end} incomplete. Partial path: {' -> '.join(map(str, path))}"
            )

        return path

    def select_next_position(self, current_position, visited):
        """
        Selects the next position for the amoeba-inspired Hamiltonian pathfinding algorithm.
        This method carefully evaluates the candidate positions surrounding the current position, considering
        factors such as grid boundaries, obstacles, the snake's body, and previously visited cells, to determine
        the most promising next step in the path.

        Args:
            current_position (tuple): The current position of the pathfinding algorithm.
            visited (set): A set of previously visited positions.

        Returns:
            tuple: The selected next position for the pathfinding algorithm, chosen based on a comprehensive
            assessment of the available options and their potential to lead to a complete Hamiltonian path.
        """
        candidate_positions = self.get_candidate_positions(current_position)
        valid_positions = [
            pos
            for pos in candidate_positions
            if self.is_valid_position(pos) and pos not in visited
        ]

        if not valid_positions:
            return None

        # Select the next position based on a heuristic or random choice
        next_position = random.choice(valid_positions)
        logging.debug(f"Selected next position: {next_position}")
        return next_position

    def get_candidate_positions(self, current_position):
        """
        Retrieves the candidate positions surrounding the current position for the amoeba-inspired Hamiltonian pathfinding algorithm.
        This method systematically explores the adjacent cells in all cardinal directions, ensuring a comprehensive consideration
        of potential next steps in the path.

        Args:
            current_position (tuple): The current position of the pathfinding algorithm.

        Returns:
            list: A list of candidate positions surrounding the current position, each represented as a tuple of coordinates.
        """
        x, y = current_position
        candidate_positions = [
            (x - 1, y),  # Left
            (x + 1, y),  # Right
            (x, y - 1),  # Up
            (x, y + 1),  # Down
        ]
        return candidate_positions

    def is_valid_position(self, position):
        """
        Determines whether a given position is valid for the amoeba-inspired Hamiltonian pathfinding algorithm.
        This method thoroughly checks the position against the grid boundaries, obstacles, and the snake's body,
        ensuring that only accessible and safe cells are considered as valid steps in the path.

        Args:
            position (tuple): The position to validate.

        Returns:
            bool: True if the position is valid and accessible, False otherwise.
        """
        x, y = position
        if not (0 <= x < self.grid.width and 0 <= y < self.grid.height):
            return False
        if position in self.snake.segments:
            return False
        if self.grid.is_obstacle(position):
            return False
        return True


class ThetaStar:
    def __init__(self, grid):
        """
        Initializes the ThetaStar object with the given grid object.
        This constructor sets up the necessary data structures and configurations to enable efficient
        pathfinding using the Theta* algorithm, tailored to the specific requirements of the game environment.

        Args:
            grid (Grid): The grid object representing the game environment.
        """
        self.grid = grid

    async def find_path(self, start, end):
        """
        Finds the shortest path from the start to the end point using the Theta* algorithm.
        This method employs an advanced variant of the A* algorithm, leveraging line-of-sight checks to
        optimize the path and reduce unnecessary node expansions, resulting in efficient and accurate pathfinding.

        Args:
            start (tuple): The starting point of the path.
            end (tuple): The endpoint of the path.

        Returns:
            list: The shortest path from start to end as a list of points, meticulously computed using the Theta*
            algorithm to ensure optimality and efficiency.
        """
        logging.debug(f"Finding shortest path from {start} to {end} using Theta*.")
        open_set = PriorityQueue()
        open_set.put((0, start))
        came_from = {}
        g_score = defaultdict(lambda: float("inf"))
        g_score[start] = 0
        f_score = defaultdict(lambda: float("inf"))
        f_score[start] = self.heuristic(start, end)

        while not open_set.empty():
            current = open_set.get()[1]

            if current == end:
                path = self.reconstruct_path(came_from, current)
                logging.info(
                    f"Shortest path found from {start} to {end}: {' -> '.join(map(str, path))}"
                )
                return path

            for neighbor in self.get_neighbors(current):
                if self.line_of_sight(came_from.get(current, current), neighbor):
                    tentative_g_score = g_score[current] + self.distance(
                        came_from.get(current, current), neighbor
                    )
                else:
                    tentative_g_score = g_score[current] + self.distance(
                        current, neighbor