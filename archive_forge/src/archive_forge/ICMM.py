class Pathfinder:
    """
    This is a universal advanced dynamic efficient pathfinding algorithm that takes in the current state of its surroundings as well as the following variables:
    - The current node coordinates
    - The nodes occuped by the rest of the body of self
    - The goal node coordinates
    - Any obstacles on the game grid/surroundings
    - The boundary of the game grid/play area/surroundings
    - Secondary goal node coordinates (optional)
    - Tertiary goal node coordinates (optional)
    - Quaternary goal node coordinates (optional)
    - Space around self
    - Space around goal(s)
    - Space around obstacles
    - Space around boundaries
    - Space around secondary goal(s) (optional)
    - Space around tertiary goal(s) (optional)
    - Space around quaternary goal(s) (optional)
    - Number of paths to generate
    - Granularity of the paths to generate
    - How frequently to update the paths
    - Escape route availability
    - Dense packing
    - Path-specific enhancements (ie zigzagging, leaving X amount of space around self, etc)
    - Body size based pathfinding changes
    - The last direction moved

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
