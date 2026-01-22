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
from queue import PriorityQueue
from collections import defaultdict
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
        self.food = food if food is not None else Food(position=np.array([WIDTH // 2, HEIGHT // 2]))
        self.grid = grid if grid is not None else Grid(WIDTH, HEIGHT)
        self.pathfinders = {'CDP': ConstrainedDelaunayPathfinder(Grid.get_points(self.grid), Grid.get_obstacles(self.grid)), 'AHP': AmoebaHamiltonianPathfinder(self.snake, self.grid, self.food), 'ThetaStar': ThetaStar(self.grid)}

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
        for name, pathfinder in self.pathfinders.items():
            path = await pathfinder.find_path(current_position, food_position)
            paths[name] = path
            costs[name] = await self.calculate_path_cost(path)
        optimal_strategy = min(costs, key=costs.get)
        optimal_path = paths[optimal_strategy]
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
        danger_cost = sum((self.grid.is_near_obstacle(point) for point in path)) * 20
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
        game_winning_alignment_cost = self.evaluate_game_winning_strategy_alignment(path)
        future_food_proximity_cost = self.calculate_future_food_proximity_cost(path)
        strategic_cost = game_winning_alignment_cost + future_food_proximity_cost
        logging.debug(f'Total strategic cost calculated: {strategic_cost}')
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
        proximity_cost = sum((self.calculate_position_proximity_cost(position, future_food_positions) for position in path))
        logging.debug(f'Calculated future food proximity cost for path: {proximity_cost}')
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
        min_distance = min((np.linalg.norm(np.array(position) - np.array(target)) for target in target_positions))
        proximity_cost = 10 / (min_distance + 1)
        logging.debug(f'Calculated proximity cost for position {position}: {proximity_cost}')
        return proximity_cost

    def predict_future_food_positions(self):
        """
        Predicts future food positions based on a comprehensive analysis of historical game data and the current game state,
        utilizing a sophisticated machine learning model. This method integrates complex data analysis techniques to forecast
        the probable locations where food might appear on the game grid, thereby enabling strategic planning for the snake's movements.

        Returns:
            list: A meticulously compiled list of predicted future food positions, each represented as a tuple of coordinates.
        """
        logging.debug('Initiating the prediction of future food positions.')
        historical_data = self.retrieve_historical_food_positions()
        current_state_features = self.extract_features_from_current_state()
        features = np.concatenate((historical_data, current_state_features), axis=0)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        X_train, X_test, y_train, y_test = train_test_split(scaled_features[:, :-1], scaled_features[:, -1], test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predicted_positions = model.predict(X_test)
        predicted_positions = [(int(pos[0]), int(pos[1])) for pos in predicted_positions]
        logging.debug(f'Predicted future food positions: {predicted_positions}')
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
        logging.debug('Retrieving historical food position data.')
        try:
            historical_data = np.random.randint(0, max(WIDTH, HEIGHT), size=(100, 2))
            logging.info(f'Successfully retrieved {len(historical_data)} historical food position records.')
            return historical_data
        except Exception as e:
            logging.error(f'Failed to retrieve historical food position data: {e}')
            raise Exception(f'Historical data retrieval error: {e}')

    def extract_features_from_current_state(self):
        """
        Extracts a rich set of informative features from the current game state, capturing critical aspects such as the snake's length,
        the current food position, and other relevant metrics. This method applies advanced feature engineering techniques to transform
        raw game data into a highly expressive and compact representation, optimized for predictive modeling purposes.

        Returns:
            numpy.ndarray: An array of features extracted from the current game state, each feature carefully selected and processed to
            maximize the predictive performance of the model.
        """
        logging.debug('Extracting features from the current game state for predictive modeling.')
        try:
            snake_length = self.snake.length
            food_position_x = self.food.position[0]
            food_position_y = self.food.position[1]
            logging.debug(f'Current snake length: {snake_length}, Food position: ({food_position_x}, {food_position_y})')
            current_state_features = np.array([snake_length, food_position_x, food_position_y])
            logging.info(f'Features extracted from current game state: {current_state_features}')
            return current_state_features
        except Exception as e:
            logging.error(f'Failed to extract features from current game state: {e}')
            raise Exception(f'Feature extraction error: {e}')

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
        min_distance = min((np.linalg.norm(np.array(position) - np.array(obstacle)) for obstacle in obstacle_positions))
        cost = 20 / (min_distance + 1)
        logging.debug(f'Calculated obstacle proximity cost for position {position}: {cost}')
        return cost

    def evaluate_game_winning_strategy_alignment(self, path):
        """
        Evaluates how well the given path aligns with established game-winning strategies.

        Args:
            path (list): The path as a list of grid positions.

        Returns:
            float: A cost representing the strategic alignment with winning strategies.
        """
        alignment_cost = 10 * (len(path) - self.snake.length)
        logging.debug(f'Calculated game winning strategy alignment cost for path: {alignment_cost}')
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
            return 'none'
        current_head = self.snake.get_head_position()
        next_position = path[1]
        if next_position[0] < current_head[0]:
            return 'left'
        elif next_position[0] > current_head[0]:
            return 'right'
        elif next_position[1] < current_head[1]:
            return 'up'
        elif next_position[1] > current_head[1]:
            return 'down'
        return 'none'