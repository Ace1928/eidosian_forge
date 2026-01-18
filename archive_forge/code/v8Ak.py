def heuristic(
        self,
        a: Tuple[int, int],
        b: Tuple[int, int],
        last_dir: Optional[Tuple[int, int]] = None,
    ) -> float:
        """
        Calculate the heuristic value for A* algorithm using the Euclidean distance.
        This heuristic is improved by adding a slight directional bias to discourage
        straight paths and promote zigzagging, which can be more optimal in certain grid setups.

        Args:
        a (Tuple[int, int]): The current node coordinates.
        b (Tuple[int, int]): The goal node coordinates.

        Returns:
        float: The computed heuristic value.
        """

        last_dir = self.direction
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        heuristic_value = dx + dy
        if last_dir:
            direction = (a[0] - b[0], a[1] - b[1])
            if direction == last_dir:
                heuristic_value += 5  # Penalize continuing in the same direction
        return heuristic_value

    def a_star_search(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Perform the A* search algorithm to find the shortest path from start to goal.
        This implementation uses a priority queue to explore the node with the lowest
        f_score and employs a heuristic that includes a directional bias to reduce path straightness.

        Args:
        start (Tuple[int, int]): The starting position of the path.
        goal (Tuple[int, int]): The goal position of the path.

        Returns:
        List[Tuple[int, int]]: The path from start to goal as a list of coordinates.
        """
        open_set = []
        heappush(open_set, (0, start))
        came_from = {start: None}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal, None)}
        last_direction = None

        while open_set:
            current = heappop(open_set)[1]
            if current == goal:
                return self.reconstruct_path(came_from, current)

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if (
                    1 <= neighbor[0] < GRID_SIZE - 1
                    and 1 <= neighbor[1] < GRID_SIZE - 1
                    and neighbor not in self.snake
                ):
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(
                            neighbor, goal, (dx, dy)
                        )
                        heappush(open_set, (f_score[neighbor], neighbor))
                        last_direction = (dx, dy)
        return []