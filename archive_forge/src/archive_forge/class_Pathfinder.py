from typing import (
import pygame  # Import pygame module for game development
from random import (
import heapq  # Import heapq module for priority queue implementation
import logging  # Import logging module for logging functionality
import math  # Import math module for mathematical operations
class Pathfinder:
    """
    A class dedicated to pathfinding within a grid-based environment using the A* algorithm.

    This class handles the computation of paths considering various environmental factors such as obstacles,
    boundaries, and the snake's own body positions. It uses the Euclidean distance for calculations and
    incorporates penalties for proximity to obstacles and boundaries.

    Attributes:
        grid_width (int): The width of the grid in which pathfinding is performed. Default is 100.
        grid_height (int): The height of the grid in which pathfinding is performed. Default is 100.
        logger (logging.Logger): Logger for recording operational logs.
        obstacles (Set[Tuple[int, int]]): A set of obstacles within the grid, represented by their positions.
    """

    def __init__(self, grid_width: int=100, grid_height: int=100, logger: Optional[logging.Logger]=None) -> None:
        """
        Initialize the Pathfinder object with the necessary dimensions and logger.

        Args:
            grid_width (int): The width of the grid. Default is 100.
            grid_height (int): The height of the grid. Default is 100.
            logger (logging.Logger): The logger object for logging messages. Default is None.
        """
        self.grid_width: int = grid_width
        self.grid_height: int = grid_height
        self.logger: logging.Logger = logger if logger is not None else logging.getLogger(__name__)

    def calculate_euclidean_distance(self, position1: Tuple[int, int], position2: Tuple[int, int]) -> float:
        """
        Calculate the Euclidean distance between two positions in the grid.

        This function utilizes basic mathematical operations to calculate the Euclidean distance between two positions.
        The Euclidean distance is more suitable for grid-based pathfinding compared to Manhattan distance in certain scenarios.

        Args:
            position1 (Tuple[int, int]): The first position tuple.
            position2 (Tuple[int, int]): The second position tuple.

        Returns:
            float: The Euclidean distance between the two positions.
        """
        dx = position2[0] - position1[0]
        dy = position2[1] - position1[1]
        return math.sqrt(dx * dx + dy * dy)

    def calculate_obstacle_proximity_penalty(self, position: Tuple[int, int], space_around_obstacles: int=5) -> float:
        """
        Calculate a penalty score based on the proximity to the nearest obstacle.

        This function iterates through each obstacle within the line of sight and calculates the distance to the given position.
        If the distance is less than the specified space around obstacles, a penalty is calculated based on the inverse of the distance.
        Closer obstacles are assigned a higher penalty to emphasize their significance.

        Args:
            position (Tuple[int, int]): The current position as a tuple.
            space_around_obstacles (int): The minimum desired distance from any obstacle. Default is 5.

        Returns:
            float: The total penalty accumulated from all nearby obstacles within the line of sight.
        """
        penalty: float = 0.0
        visible_obstacles: Set[Tuple[int, int]] = self.get_line_of_sight_obstacles(position)
        for obstacle in visible_obstacles:
            distance: float = self.calculate_euclidean_distance(position, obstacle)
            if distance <= space_around_obstacles:
                penalty += 1 / (distance + 1)
        return penalty

    def calculate_boundary_proximity_penalty(self, position: Tuple[int, int], boundaries: Tuple[int, int, int, int]=(0, 0, 100, 100), space_around_boundaries: int=5) -> float:
        """
        Calculate a penalty based on the proximity to boundaries.

        This function computes a penalty score based on how close the given position is to the boundaries of the environment.
        The penalty increases as the position approaches the boundary within a specified threshold.

        Args:
            position (Tuple[int, int]): The current position as a tuple.
            boundaries (Tuple[int, int, int, int]): The boundaries of the environment as a tuple of (x_min, y_min, x_max, y_max). Default is (0, 0, 100, 100).
            space_around_boundaries (int): The desired space to maintain around boundaries. Default is 5.

        Returns:
            float: The calculated penalty based on proximity to boundaries.
        """
        x_min, y_min, x_max, y_max = boundaries
        min_distance_to_boundary: float = min(position[0] - x_min, x_max - position[0], position[1] - y_min, y_max - position[1])
        if min_distance_to_boundary < space_around_boundaries:
            return (space_around_boundaries - min_distance_to_boundary) ** 2
        return 0.0

    def calculate_body_position_proximity_penalty(self, position: Tuple[int, int], body_positions: Set[Tuple[int, int]], space_around_agent: int=2) -> float:
        """
        Calculate a penalty for being too close to the snake's own body.

        This function iterates through each position occupied by the snake's body within the line of sight
        and calculates a penalty if the given position is within a specified distance from any part of the body.
        The penalty is set to infinity to represent an impassable barrier.

        Args:
            position (Tuple[int, int]): The current position as a tuple.
            body_positions (Set[Tuple[int, int]]): The positions occupied by the snake's body.
            space_around_agent (int): The desired space to maintain around the snake's body. Default is 2.

        Returns:
            float: The calculated penalty for being too close to the snake's body.
        """
        penalty: float = 0.0
        visible_body_positions: Set[Tuple[int, int]] = self.get_line_of_sight_body_positions(position)
        for body_position in visible_body_positions:
            if self.calculate_euclidean_distance(position, body_position) < space_around_agent:
                penalty += float('inf')
        return penalty

    def evaluate_escape_routes(self, position: Tuple[int, int], obstacles: Set[Tuple[int, int]], boundaries: Tuple[int, int, int, int]=(0, 0, 100, 100)) -> float:
        """
        Evaluate and score the availability of escape routes.

        This function assesses the number of available escape routes from the current position.
        It checks each cardinal direction (up, down, left, right) and scores based on the number of unobstructed paths.

        Args:
            position (Tuple[int, int]): The current position as a tuple.
            obstacles (Set[Tuple[int, int]]): The positions of obstacles in the environment.
            boundaries (Tuple[int, int, int, int]): The boundaries of the environment as a tuple of (x_min, y_min, x_max, y_max). Default is (0, 0, 100, 100).

        Returns:
            float: The score based on the availability of escape routes.
        """
        score: float = 0.0
        directions: List[Tuple[int, int]] = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for direction in directions:
            neighbor: Tuple[int, int] = (position[0] + direction[0], position[1] + direction[1])
            if neighbor not in obstacles and self.is_position_within_boundaries(neighbor, boundaries):
                score += 1.0
        return -score

    def is_position_within_boundaries(self, position: Tuple[int, int], boundaries: Tuple[int, int, int, int]=(0, 0, 100, 100)) -> bool:
        """
        Check if a position is within the specified boundaries.

        This function determines whether a given position falls within the defined environmental boundaries.

        Args:
            position (Tuple[int, int]): The position to check as a tuple.
            boundaries (Tuple[int, int, int, int]): The boundaries of the environment as a tuple of (x_min, y_min, x_max, y_max). Default is (0, 0, 100, 100).

        Returns:
            bool: True if the position is within the boundaries, False otherwise.
        """
        x_min, y_min, x_max, y_max = boundaries
        return x_min <= position[0] <= x_max and y_min <= position[1] <= y_max

    def apply_zigzagging_effect(self, current_heuristic: float=1.0) -> float:
        """
        Modify the heuristic to account for zigzagging, making the path less predictable.

        This function increases the heuristic value slightly to account for the added complexity of zigzagging,
        which can make the path less predictable and potentially safer from pursuers.

        Args:
            current_heuristic (float): The current heuristic value. Default is 1.0.

        Returns:
            float: The modified heuristic value accounting for zigzagging.
        """
        return current_heuristic * 1.05

    def apply_dense_packing_effect(self, current_heuristic: float=1.0) -> float:
        """
        Modify the heuristic to handle dense packing scenarios more effectively.

        This function decreases the heuristic value to account for dense packing scenarios,
        where closer packing might be necessary or unavoidable.

        Args:
            current_heuristic (float): The current heuristic value. Default is 1.0.

        Returns:
            float: The modified heuristic value accounting for dense packing.
        """
        return current_heuristic * 0.95

    def get_line_of_sight_obstacles(self, position: Tuple[int, int], sight_range: int=5) -> Set[Tuple[int, int]]:
        """
        Dynamically calculate obstacles within the line of sight of the agent.

        This function determines the obstacles that are visible to the agent based on its current position and sight range.
        It checks the surrounding positions within the sight range and adds any obstacles found to the set of visible obstacles.

        Args:
            position (Tuple[int, int]): The current position of the agent as a tuple.
            sight_range (int): The range of sight for the agent. Default is 5.

        Returns:
            Set[Tuple[int, int]]: A set of obstacle positions within the line of sight of the agent.
        """
        visible_obstacles: Set[Tuple[int, int]] = set()
        for x in range(max(0, position[0] - sight_range), min(self.grid_width, position[0] + sight_range + 1)):
            for y in range(max(0, position[1] - sight_range), min(self.grid_height, position[1] + sight_range + 1)):
                current_position: Tuple[int, int] = (x, y)
                if current_position == Pathfinder.astar_search.goal_position or current_position in self.body_positions:
                    visible_obstacles.add(current_position)
        return visible_obstacles

    def get_line_of_sight_body_positions(self, position: Tuple[int, int], sight_range: int=5) -> Set[Tuple[int, int]]:
        """
        Dynamically calculate the snake's body positions within the line of sight of the agent.

        This function determines the positions occupied by the snake's body that are visible to the agent based on its current position and sight range.
        It checks the surrounding positions within the sight range and adds any body positions found to the set of visible body positions.

        Args:
            position (Tuple[int, int]): The current position of the agent as a tuple.
            sight_range (int): The range of sight for the agent. Default is 5.

        Returns:
            Set[Tuple[int, int]]: A set of body positions within the line of sight of the agent.
        """
        visible_body_positions: Set[Tuple[int, int]] = set()
        for x in range(max(0, position[0] - sight_range), min(self.grid_width, position[0] + sight_range + 1)):
            for y in range(max(0, position[1] - sight_range), min(self.grid_height, position[1] + sight_range + 1)):
                current_position: Tuple[int, int] = (x, y)
                if self.is_position_body(current_position):
                    visible_body_positions.add(current_position)
        return visible_body_positions

    def get_line_of_sight_goals(self, position: Tuple[int, int], sight_range: int=5) -> Set[Tuple[int, int]]:
        """
        Dynamically calculate the goal positions within the line of sight of the agent.

        This function determines the goal positions that are visible to the agent based on its current position and sight range.
        It checks the surrounding positions within the sight range and adds any goal positions found to the set of visible goals.

        Args:
            position (Tuple[int, int]): The current position of the agent as a tuple.
            sight_range (int): The range of sight for the agent. Default is 5.

        Returns:
            Set[Tuple[int, int]]: A set of goal positions within the line of sight of the agent.
        """
        visible_goal_positions: Set[Tuple[int, int]] = set()
        for x in range(max(0, position[0] - sight_range), min(self.grid_width, position[0] + sight_range + 1)):
            for y in range(max(0, position[1] - sight_range), min(self.grid_height, position[1] + sight_range + 1)):
                current_position: Tuple[int, int] = (x, y)
                if current_position in self.astar_search.goal_positions:
                    visible_goal_positions.add(current_position)
        return visible_goal_positions

    def heuristic(self, current_position: Tuple[int, int], goal_position: Tuple[int, int], secondary_goal_position: Tuple[int, int]=None, tertiary_goal_position: Tuple[int, int]=None, quaternary_goal_position: Tuple[int, int]=None, environment_boundaries: Tuple[int, int, int, int]=(0, 0, 100, 100), space_around_agent: int=0, space_around_goals: int=0, space_around_obstacles: int=0, space_around_boundaries: int=0, obstacles: Set[Tuple[int, int]]=set(), escape_route_availability: bool=False, enhancements: List[str]=None, dense_packing: bool=True, body_size_adaptations: bool=True, self_body_positions: Set[Tuple[int, int]]=set()) -> float:
        """
        Calculate the heuristic value for the Dynamic Pathfinding algorithm.

        This heuristic function incorporates multiple factors to determine the estimated cost
        from the current position to the goal position. It takes into account the primary goal
        position, as well as optional secondary, tertiary, and quaternary goal positions.
        The heuristic value is adjusted based on the proximity to obstacles, boundaries, and
        the agent's own body positions. It also considers factors such as escape route availability,
        dense packing scenarios, and body size adaptations.

        Args:
            current_position (Tuple[int, int]): The current position of the agent.
            goal_position (Tuple[int, int]): The primary target position the agent aims to reach.
            secondary_goal_position (Tuple[int, int], optional): Secondary target position. Defaults to None.
            tertiary_goal_position (Tuple[int, int], optional): Tertiary target position. Defaults to None.
            quaternary_goal_position (Tuple[int, int], optional): Quaternary target position. Defaults to None.
            environment_boundaries (Tuple[int, int, int, int], optional): The boundaries of the environment.
                Defaults to (0, 0, 100, 100).
            space_around_agent (int, optional): The desired space to maintain around the agent. Defaults to 0.
            space_around_goals (int, optional): The desired space to maintain around goal positions. Defaults to 0.
            space_around_obstacles (int, optional): The desired space to maintain around obstacles. Defaults to 0.
            space_around_boundaries (int, optional): The desired space to maintain around boundaries. Defaults to 0.
            obstacles (Set[Tuple[int, int]], optional): The positions of obstacles in the environment. Defaults to set().
            escape_route_availability (bool, optional): Flag indicating whether escape routes should be considered. Defaults to False.
            enhancements (List[str], optional): List of enhancements to apply to the heuristic calculation. Defaults to None.
            dense_packing (bool, optional): Flag indicating whether dense packing scenarios should be considered. Defaults to True.
            body_size_adaptations (bool, optional): Flag indicating whether body size adaptations should be considered. Defaults to True.
            self_body_positions (Set[Tuple[int, int]], optional): The positions occupied by the agent's own body. Defaults to set().

        Returns:
            float: The calculated heuristic value estimating the cost from the current position to the goal position.
        """
        heuristic_value: float = 0.0
        primary_goal_distance: float = self.calculate_euclidean_distance(current_position, goal_position)
        heuristic_value += primary_goal_distance
        if secondary_goal_position is not None:
            secondary_goal_distance: float = self.calculate_euclidean_distance(current_position, secondary_goal_position)
            heuristic_value += 0.5 * secondary_goal_distance
        if tertiary_goal_position is not None:
            tertiary_goal_distance: float = self.calculate_euclidean_distance(current_position, tertiary_goal_position)
            heuristic_value += 0.3 * tertiary_goal_distance
        if quaternary_goal_position is not None:
            quaternary_goal_distance: float = self.calculate_euclidean_distance(current_position, quaternary_goal_position)
            heuristic_value += 0.1 * quaternary_goal_distance
        obstacle_proximity_penalty: float = self.calculate_obstacle_proximity_penalty(current_position, space_around_obstacles)
        heuristic_value += obstacle_proximity_penalty
        boundary_proximity_penalty: float = self.calculate_boundary_proximity_penalty(current_position, environment_boundaries, space_around_boundaries)
        heuristic_value += boundary_proximity_penalty
        if body_size_adaptations:
            body_position_proximity_penalty: float = self.calculate_body_position_proximity_penalty(current_position, self_body_positions, space_around_agent)
            heuristic_value += body_position_proximity_penalty
        if escape_route_availability:
            escape_route_score: float = self.evaluate_escape_routes(current_position, obstacles, environment_boundaries)
            heuristic_value += escape_route_score
        if enhancements is not None:
            for enhancement in enhancements:
                if enhancement == 'zigzagging':
                    heuristic_value = self.apply_zigzagging_effect(heuristic_value)
                elif enhancement == 'dense_packing':
                    heuristic_value = self.apply_dense_packing_effect(heuristic_value)
        self.logger.debug(f'Calculated heuristic value: {heuristic_value}')
        return heuristic_value

    def astar_search(self, start_position: Tuple[int, int], goal_position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Implement the A* search algorithm to find the optimal path from start to goal.

        This function utilizes the A* search algorithm to find the optimal path from the start position to the goal position.
        It maintains an open set of positions to explore, and a closed set of visited positions.
        The algorithm selects the position with the lowest total cost (g_cost + h_cost) from the open set,
        explores its neighbors, and updates the costs and paths accordingly.
        It continues until the goal position is reached or there are no more positions to explore.
        The path is then extended to create a full cycle, zigzagging from the goal back to the start position.

        Args:
            start_position (Tuple[int, int]): The starting position of the search.
            goal_position (Tuple[int, int]): The target position to reach.

        Returns:
            List[Tuple[int, int]]: The optimal path from start to goal and back to start as a list of positions,
                or an empty list if no path is found.
        """
        open_set: List[Tuple[float, float, Tuple[int, int]]] = [(0, 0, start_position)]
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start_position: None}
        g_cost: Dict[Tuple[int, int], float] = {start_position: 0}
        while open_set:
            _, current_g_cost, current_position = heapq.heappop(open_set)
            if current_position == goal_position:
                path_to_goal: List[Tuple[int, int]] = self.reconstruct_path(came_from, start_position, goal_position)
                path_to_start: List[Tuple[int, int]] = self.reconstruct_path(came_from, goal_position, start_position)
                path_to_start.reverse()
                full_path: List[Tuple[int, int]] = path_to_goal + path_to_start
                return full_path
            for neighbor_position in self.get_neighbors(current_position):
                tentative_g_cost: float = current_g_cost + self.calculate_euclidean_distance(current_position, neighbor_position)
                if neighbor_position not in g_cost or tentative_g_cost < g_cost[neighbor_position]:
                    g_cost[neighbor_position] = tentative_g_cost
                    h_cost: float = self.heuristic(neighbor_position, goal_position)
                    total_cost: float = tentative_g_cost + h_cost
                    heapq.heappush(open_set, (total_cost, tentative_g_cost, neighbor_position))
                    came_from[neighbor_position] = current_position
        return []

    def reconstruct_path(self, came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]], start_position: Tuple[int, int], goal_position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct the optimal path from start to goal using the came_from dictionary.

        This function takes the came_from dictionary, start position, and goal position,
        and reconstructs the optimal path from the start to the goal by backtracking through the came_from dictionary.
        It starts from the goal position and follows the parent positions until reaching the start position.
        The reconstructed path is then reversed to obtain the correct order from start to goal.

        Args:
            came_from (Dict[Tuple[int, int], Optional[Tuple[int, int]]]): A dictionary mapping each position to its parent position in the optimal path.
            start_position (Tuple[int, int]): The starting position of the path.
            goal_position (Tuple[int, int]): The target position of the path.

        Returns:
            List[Tuple[int, int]]: The reconstructed optimal path from start to goal as a list of positions.
        """
        path: List[Tuple[int, int]] = []
        current_position: Tuple[int, int] = goal_position
        while current_position != start_position:
            path.append(current_position)
            current_position = came_from[current_position]
        path.append(start_position)
        path.reverse()
        return path

    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get the valid neighboring positions of a given position.

        This function takes a position and returns a list of its valid neighboring positions.
        The neighboring positions are calculated by adding the four cardinal directions (up, right, down, left) to the current position.
        The validity of each neighboring position is checked against the grid boundaries and obstacle positions.
        Only the positions that are within the grid boundaries and not occupied by obstacles are considered valid neighbors.

        Args:
            position (Tuple[int, int]): The position for which to get the neighbors.

        Returns:
            List[Tuple[int, int]]: A list of valid neighboring positions.
        """
        directions: List[Tuple[int, int]] = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        neighbors: List[Tuple[int, int]] = []
        for direction in directions:
            new_position: Tuple[int, int] = (position[0] + direction[0], position[1] + direction[1])
            if self.is_position_within_boundaries(new_position):
                if not self.get_line_of_sight_obstacles(new_position):
                    neighbors.append(new_position)
        return neighbors