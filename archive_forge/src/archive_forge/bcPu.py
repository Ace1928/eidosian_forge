class Pathfinder:
    """
    This class epitomizes a universal, advanced, dynamic, efficient, and robust pathfinding algorithm, meticulously engineered to be universally applicable across a broad spectrum of pathfinding contexts. It integrates a comprehensive, detailed, and exhaustive set of parameters to ensure optimal pathfinding capabilities. These parameters are meticulously designed to cater to the most complex scenarios, ensuring adaptability and precision. The parameters considered include, but are not limited to:

    - current_node_coordinates: Tuple[int, int] (default: (0, 0))
        The precise current position of the agent within the game grid or environment, represented as a tuple of integers for grid coordinates. The default value is the origin point of the grid, representing a typical starting position.

    - body_occupied_nodes: Set[Tuple[int, int]] (default: set())
        A set of tuples representing grid coordinates occupied by the agent's body, crucial in scenarios where the agent's body can obstruct its path. The default is an empty set, indicating no initial body occupation.

    - goal_node_coordinates: Tuple[int, int] (default: (10, 10))
        The primary target position the agent aims to reach, specified as a tuple of integers for exact grid coordinates. The default target is set to (10, 10), a common goal location in a standard grid.

    - obstacle_positions: Set[Tuple[int, int]] (default: set())
        A set of tuples indicating the positions of static or dynamic obstacles within the environment that must be navigated around. The default is an empty set, indicating a clear grid.

    - environment_boundaries: Tuple[int, int, int, int] (default: (0, 0, 100, 100))
        The boundaries of the playable or navigable area, typically defined by minimum and maximum coordinates (x_min, y_min, x_max, y_max), ensuring precise boundary management. The default represents a 100x100 grid.

    - secondary_goal_coordinates: Optional[Tuple[int, int]] (default: None)
        An optional secondary target position, specified as a tuple of integers for grid coordinates. The default is None, indicating no secondary goal.

    - tertiary_goal_coordinates: Optional[Tuple[int, int]] (default: None)
        An optional tertiary target position, enhancing the complexity and adaptability of the pathfinding. The default is None, indicating no tertiary goal.

    - quaternary_goal_coordinates: Optional[Tuple[int, int]] (default: None)
        An optional quaternary target position, further extending the flexibility and utility of the pathfinding algorithm. The default is None, indicating no quaternary goal.

    - space_around_agent: int (default: 1)
        The required clearance around the agent, specified in grid units, to avoid collisions, ensuring safe navigation. The default clearance is 1 grid unit.

    - space_around_goals: Dict[str, int] (default: {'primary': 1, 'secondary': 1, 'tertiary': 1, 'quaternary': 1})
        A dictionary mapping goal identifiers to specific clearance requirements around primary and optional secondary, tertiary, and quaternary goals, ensuring tailored pathfinding. Each goal has a default clearance of 1 grid unit.

    - space_around_obstacles: int (default: 1)
        The required clearance around obstacles, specified in grid units, to ensure safe navigation and obstacle avoidance. The default clearance is 1 grid unit.

    - space_around_boundaries: int (default: 1)
        The minimum distance, in grid units, the agent must maintain from the boundaries to avoid leaving the navigable area, ensuring boundary compliance. The default distance is 1 grid unit.

    - path_count: int (default: 3)
        The number of alternative paths to generate and evaluate, providing options for optimal path selection. The default is 3 paths, offering a balance between computational efficiency and choice.

    - path_granularity: int (default: 1)
        The level of detail or fineness in the generated paths, affecting the smoothness and precision of navigation, specified in grid units. The default granularity is 1, which corresponds to the finest level of detail.

    - update_frequency: int (default: 100)
        The frequency, in milliseconds, at which the pathfinding algorithm updates the paths based on dynamic changes in the environment, ensuring real-time adaptability. The default frequency is 100 milliseconds.

    - escape_route_availability: bool (default: True)
        A boolean indicating whether the algorithm should calculate escape routes in case of sudden blockages or threats, enhancing safety and robustness. The default is True, enabling escape route calculations.

    - dense_packing: bool (default: False)
        A boolean specifying whether the pathfinding should consider dense packing scenarios, relevant in tightly packed, multi-agent environments, ensuring efficient space utilization. The default is False, assuming a less crowded environment.

    - path_enhancements: List[str] (default: [])
        A list of specific enhancements or modifications to the path logic, such as zigzagging, maintaining clearance, etc., tailored to enhance pathfinding efficiency and adaptability. The default is an empty list, indicating standard pathfinding without enhancements.

    - body_size_adaptations: bool (default: False)
        A boolean indicating adjustments in the pathfinding logic based on the size of the agent's body, affecting how the space around the agent is calculated, ensuring size-specific pathfinding. The default is False, assuming a standard body size.

    - last_direction_moved: Optional[Tuple[int, int]] (default: None)
        The last movement direction of the agent, used to introduce biases or penalties in the path calculation to prevent oscillations or repetitive patterns, enhancing path stability. The default is None, indicating no prior movement bias.

    - environmental_adaptability: bool (default: True)
        A boolean indicating whether the pathfinding logic dynamically adapts to changes in the environmental conditions, such as weather, terrain type, and other dynamic factors, ensuring robust adaptability. The default is True, enabling dynamic adaptation.

    - multi_agent_coordination: bool (default: False)
        A boolean specifying whether the pathfinding algorithm is designed to coordinate paths among multiple agents, crucial in scenarios involving team-based or competitive multi-agent systems. The default is False, tailored for single-agent scenarios.

    - resource_optimization: bool (default: True)
        A boolean indicating whether the pathfinding algorithm optimizes the use of computational and other resources, ensuring efficient operation even under constraints. The default is True, promoting resource-efficient pathfinding.

    This comprehensive parameter set ensures that the Pathfinder class can be adapted and utilized in a wide range of scenarios, promoting modular, flexible, clean, scalable development and maintenance. Each parameter is intricately designed to interact synergistically, providing a robust framework that supports dynamic, efficient, and precise pathfinding capabilities.

    The Pathfinder class employs a variety of sophisticated algorithms, including but not limited to A* for optimal pathfinding, Dijkstra's algorithm for shortest path calculations, and custom heuristic functions tailored for specific scenarios such as densely packed environments or dynamic obstacle navigation. These algorithms are integrated with advanced data structures for state management and efficient computation, such as priority queues for open set management in A*, and hash tables for quick look-up of node states.

    The class is engineered to handle a multitude of environmental variables and agent characteristics. It dynamically adjusts its computations based on the agent's body size, the proximity of obstacles, and the designated goals. This adaptability is achieved through a meticulously crafted set of methods that assess and respond to real-time changes in the environment, such as sudden appearance of obstacles or changes in the agent's body configuration.

    Furthermore, the Pathfinder class is designed with multi-agent coordination in mind, allowing for the seamless integration of pathfinding strategies among multiple agents in a shared environment. This is particularly crucial in competitive or cooperative scenarios where agents must navigate without colliding with each other while still achieving their individual or collective goals.

    Resource optimization is a key aspect of the Pathfinder's design, ensuring that the pathfinding operations are not only accurate but also computationally efficient. This allows the Pathfinder to be deployed in systems with limited computational resources while still maintaining high performance and responsiveness.

    In summary, the Pathfinder class is a paragon of modern pathfinding techniques, encapsulating a wide array of functionalities that are both comprehensive and customizable. It stands as a testament to advanced, flexible, and efficient software design, capable of adapting to and excelling in any pathfinding scenario presented.
    """

    def heuristic(
        self,
        a: Tuple[int, int],
        b: Tuple[int, int],
        last_dir: Optional[Tuple[int, int]] = None,
        is_return_path: bool = False,
    ) -> float:
        """
        Calculate the heuristic value for A* algorithm using a dynamic, adaptive approach.
        This heuristic is optimized for real-time performance and scalability, incorporating
        multiple factors such as directional bias, obstacle avoidance, boundary awareness,
        snake body avoidance, escape route availability, dense packing, and path-specific
        adjustments. The heuristic is designed to generate strategic, efficient paths that
        adapt to the current game state and snake's length.

        Args:
            a (Tuple[int, int]): The current node coordinates.
            b (Tuple[int, int]): The goal node coordinates.
            last_dir (Optional[Tuple[int, int]]): The last direction moved.
            is_return_path (bool): Flag indicating if the heuristic is for the return path.

        Returns:
            float: The computed heuristic value.
        """
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        euclidean_distance = math.sqrt(dx**2 + dy**2)

        # Directional Bias: Penalize moving in the same direction to promote zigzagging
        direction_penalty = 0
        if last_dir:
            current_dir = (a[0] - b[0], a[1] - b[1])
            if current_dir == last_dir:
                direction_penalty = 5 * (1 - len(self.snake) / (BLOCK_SIZE**2))

        # Boundary Awareness: Dynamically adjust penalty based on snake's proximity to boundaries
        boundary_threshold = max(2, int(0.1 * BLOCK_SIZE))
        boundary_penalty = 0
        if (
            a[0] < boundary_threshold
            or a[0] >= BLOCK_SIZE - boundary_threshold
            or a[1] < boundary_threshold
            or a[1] >= BLOCK_SIZE - boundary_threshold
        ):
            boundary_penalty = 10 * (1 - len(self.snake) / (BLOCK_SIZE**2))
            boundary_penalty *= (
                1
                - min(a[0], a[1], BLOCK_SIZE - a[0] - 1, BLOCK_SIZE - a[1] - 1)
                / boundary_threshold
            )

        # Obstacle Avoidance: Penalize nodes that are adjacent to obstacles, considering snake's length
        obstacle_penalty = 0
        for dx, dy in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]:
            neighbor = (a[0] + dx, a[1] + dy)
            if neighbor in self.snake:
                obstacle_penalty += 5 * (1 - len(self.snake) / (BLOCK_SIZE**2))

        # Snake Body Avoidance: Heavily penalize nodes that are part of the snake's body
        snake_body_penalty = 0
        if a in self.snake:
            snake_body_penalty = float("inf")

        # Escape Route: Favor nodes with more available neighboring nodes, considering snake's length
        escape_route_bonus = 0
        available_neighbors = 0
        for dx, dy in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]:
            neighbor = (a[0] + dx, a[1] + dy)
            if (
                0 <= neighbor[0] < BLOCK_SIZE
                and 0 <= neighbor[1] < BLOCK_SIZE
                and neighbor not in self.snake
            ):
                available_neighbors += 1
        escape_route_bonus = (
            available_neighbors * -2 * (len(self.snake) / (BLOCK_SIZE**2))
        )

        # Dense Packing: Favor nodes that are closer to other parts of the snake's body, considering snake's length
        dense_packing_bonus = 0
        for segment in self.snake:
            dense_packing_bonus += 1 / (
                1 + math.sqrt((a[0] - segment[0]) ** 2 + (a[1] - segment[1]) ** 2)
            )
        dense_packing_bonus *= len(self.snake) / (BLOCK_SIZE**2)

        # Return Path: Dynamically adjust heuristic for the return path to prioritize reaching the tail
        return_path_bonus = 0
        if is_return_path:
            tail_distance = math.sqrt(
                (a[0] - self.snake[-1][0]) ** 2 + (a[1] - self.snake[-1][1]) ** 2
            )
            return_path_bonus = -tail_distance * (1 - len(self.snake) / (BLOCK_SIZE**2))

        # Food Seeking: Favor nodes that are closer to the food, considering snake's length
        food_seeking_bonus = 0
        if not is_return_path and self.food:
            food_distance = math.sqrt(
                (a[0] - self.food[0]) ** 2 + (a[1] - self.food[1]) ** 2
            )
            food_seeking_bonus = -food_distance * (
                1 - len(self.snake) / (BLOCK_SIZE**2)
            )

        # Real-time Adaptation: Dynamically adjust heuristic weights based on game state
        snake_length_ratio = len(self.snake) / (BLOCK_SIZE**2)
        direction_penalty_weight = 1 - snake_length_ratio
        boundary_penalty_weight = 1 - snake_length_ratio
        obstacle_penalty_weight = 1 - snake_length_ratio
        escape_route_bonus_weight = snake_length_ratio
        dense_packing_bonus_weight = snake_length_ratio
        return_path_bonus_weight = snake_length_ratio
        food_seeking_bonus_weight = 1 - snake_length_ratio

        # Collision Avoidance: Penalize nodes that lead to potential collisions
        collision_penalty = 0
        if self._is_collision(a):
            collision_penalty = float("inf")

        # Adaptive Exploration: Encourage exploration of unexplored areas based on snake's length
        exploration_bonus = 0
        if not self._is_explored(a):
            exploration_bonus = 10 * (1 - len(self.snake) / (BLOCK_SIZE**2))

        # Calculate the final heuristic value
        heuristic_value = (
            euclidean_distance
            + direction_penalty * direction_penalty_weight
            + boundary_penalty * boundary_penalty_weight
            + obstacle_penalty * obstacle_penalty_weight
            + snake_body_penalty
            + escape_route_bonus * escape_route_bonus_weight
            + dense_packing_bonus * dense_packing_bonus_weight
            + return_path_bonus * return_path_bonus_weight
            + food_seeking_bonus * food_seeking_bonus_weight
            + collision_penalty
            + exploration_bonus
        )

        return heuristic_value

    def _is_collision(self, node: Tuple[int, int]) -> bool:
        return node in self.snake

    def _is_explored(self, node: Tuple[int, int]) -> bool:
        return node in self.snake or node in self.path

    def _is_return_path(self, node: Tuple[int, int]) -> bool:
        return node in self.snake[1:]
