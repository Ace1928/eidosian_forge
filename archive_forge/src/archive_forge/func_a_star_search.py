def a_star_search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Perform the A* search algorithm to find the optimal path from start to goal and back to the tail.
    This implementation uses a priority queue to efficiently explore nodes, a closed set
    to avoid redundant processing, and a custom heuristic function that considers multiple
    factors to generate strategic and efficient paths. It also calculates a complete cycle by
    finding the path to the goal and then the path from the goal back to the snake's tail.
    The search is optimized for real-time performance, scalability, and adaptability to the
    current game state.

    Args:
        start (Tuple[int, int]): The starting position of the path.
        goal (Tuple[int, int]): The goal position of the path.

    Returns:
        List[Tuple[int, int]]: The optimal path from start to goal and back to the tail as a list of coordinates.
    """
    path_to_goal = self._a_star_search(start, goal, is_return_path=False)
    if not path_to_goal:
        return []
    path_to_tail = self._a_star_search(path_to_goal[-1], self.snake[-1], is_return_path=True)
    complete_path = path_to_goal + path_to_tail
    optimized_path = self._optimize_path(complete_path)
    collision_free_path = self._avoid_collisions(optimized_path)
    return collision_free_path