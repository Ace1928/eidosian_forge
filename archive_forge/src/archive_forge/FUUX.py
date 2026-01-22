import pygame
import random
import logging
import math
from heapq import heappush, heappop
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
from queue import PriorityQueue

# Initialize logging
logging.basicConfig(
    filename="pygame_snake_game.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Define colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
PINK = (255, 192, 203)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)

# Game configuration
CELL_SIZE = 20
GRID_SIZE = 20
FPS = 60
GAME_TICK = 10
BLOCK_SIZE = GRID_SIZE  # Standardized naming for grid size
BEZIER_RESOLUTION = 10  # Resolution for Bezier curve interpolation
OPEN_SET_LIMIT = 100  # Limit for open set pruning in A* search


class SnakeGameAI:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(
            (GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE)
        )
        pygame.display.set_caption("Snake Game AI")
        self.clock = pygame.time.Clock()
        self.reset_game()

    def reset_game(self) -> None:
        self.snake: List[Tuple[int, int]] = [(GRID_SIZE // 2, GRID_SIZE // 2)]
        self.score: int = 0
        self.food: Optional[Tuple[int, int]] = None
        self.game_over: bool = False
        self.direction: Tuple[int, int] = (1, 0)  # Start moving right
        self.path: List[Tuple[int, int]] = []
        self.tail_path: List[Tuple[int, int]] = []
        self.place_food()
        logging.info("Game reset.")

    def place_food(self) -> None:
        while self.food is None:
            potential_food = (
                random.randint(1, GRID_SIZE - 2),
                random.randint(1, GRID_SIZE - 2),
            )
            if potential_food not in self.snake:
                self.food = potential_food
                logging.debug(f"Food placed at {self.food}")

    def play_step(self) -> bool:
        if self.game_over:
            return self.game_over
        self.move_snake()
        self.check_collision()
        return self.game_over

    def move_snake(self) -> None:
        head_x, head_y = self.snake[0]
        new_x = head_x + self.direction[0]
        new_y = head_y + self.direction[1]
        new_head = (new_x, new_y)
        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            self.place_food()
            self.path = []
            self.tail_path = []
        else:
            self.snake.pop()

    def check_collision(self) -> None:
        head = self.snake[0]
        if (
            head in self.snake[1:]
            or head[0] < 1
            or head[0] >= GRID_SIZE - 1
            or head[1] < 1
            or head[1] >= GRID_SIZE - 1
        ):
            self.game_over = True
            logging.error("Collision detected: Game over.")

    def draw_elements(self) -> None:
        self.screen.fill(BLACK)
        # Draw borders
        pygame.draw.rect(self.screen, WHITE, [0, 0, GRID_SIZE * CELL_SIZE, CELL_SIZE])
        pygame.draw.rect(
            self.screen,
            WHITE,
            [0, (GRID_SIZE - 1) * CELL_SIZE, GRID_SIZE * CELL_SIZE, CELL_SIZE],
        )
        pygame.draw.rect(self.screen, WHITE, [0, 0, CELL_SIZE, GRID_SIZE * CELL_SIZE])
        pygame.draw.rect(
            self.screen,
            WHITE,
            [(GRID_SIZE - 1) * CELL_SIZE, 0, CELL_SIZE, GRID_SIZE * CELL_SIZE],
        )
        # Draw path
        for x, y in self.path + self.tail_path:
            pygame.draw.rect(
                self.screen,
                YELLOW,
                [x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE],
                2,
            )
        # Draw snake
        for i, (x, y) in enumerate(self.snake):
            color = ((i * 50) % 255, (i * 100) % 255, (i * 150) % 255)
            pygame.draw.rect(
                self.screen, color, [x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE]
            )
        # Draw food
        if self.food:
            color = RED if pygame.time.get_ticks() % 500 < 250 else PINK
            pygame.draw.rect(
                self.screen,
                color,
                [
                    self.food[0] * CELL_SIZE,
                    self.food[1] * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE,
                ],
            )
        pygame.display.update()

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

    def a_star_search(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
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
        if depth > 10:  # Limit recursion depth to prevent infinite recursion
            return []
        path_to_goal = self.a_star_search_helper(start, goal, is_return_path=False)
        if not path_to_goal:
            return []

        # Find the path from goal to the snake's tail
        path_to_tail = self.a_star_search_helper(
            path_to_goal[-1], self.snake[-1], is_return_path=True
        )
        complete_path = path_to_goal + path_to_tail
        optimized_path = self._optimize_path(complete_path)
        collision_free_path = self.avoid_collisions(
            optimized_path, depth + 1
        )  # Increment depth
        return collision_free_path

    def a_star_search_helper(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        is_return_path: bool = False,
    ) -> List[Tuple[int, int]]:
        """
        Helper function to perform the A* search algorithm for a single path.
        This implementation is optimized for real-time performance and scalability, using
        efficient data structures and pruning techniques to minimize the search space.
        It also incorporates adaptive heuristics and dynamic adjustments based on the current
        game state to generate strategic and efficient paths.

        Args:
            start (Tuple[int, int]): The starting position of the path.
            goal (Tuple[int, int]): The goal position of the path.
            is_return_path (bool): Flag indicating if the search is for the return path.

        Returns:
            List[Tuple[int, int]]: The optimal path from start to goal as a list of coordinates.
        """
        open_set = PriorityQueue()
        open_set.put((0, start))
        came_from = {start: None}
        g_score = defaultdict(lambda: float("inf"))
        g_score[start] = 0
        f_score = defaultdict(lambda: float("inf"))
        f_score[start] = self.heuristic(start, goal, is_return_path=is_return_path)

        closed_set = set()
        last_direction = None

        while not open_set.empty():
            current = open_set.get()[1]
            closed_set.add(current)

            if current == goal:
                return self.reconstruct_path(came_from, current)

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
                neighbor = (current[0] + dx, current[1] + dy)
                if (
                    0 <= neighbor[0] < BLOCK_SIZE
                    and 0 <= neighbor[1] < BLOCK_SIZE
                    and neighbor not in self.snake
                    and neighbor not in closed_set
                ):
                    tentative_g_score = g_score[current] + math.sqrt(dx**2 + dy**2)
                    if tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(
                            neighbor, goal, (dx, dy), is_return_path
                        )
                        open_set.put((f_score[neighbor], neighbor))
                        last_direction = (dx, dy)

            # Prune the open set to remove low-priority nodes and maintain a manageable search space
            if open_set.qsize() > OPEN_SET_LIMIT:
                pruned_open_set = PriorityQueue()
                for _ in range(OPEN_SET_LIMIT):
                    pruned_open_set.put(open_set.get())
                open_set = pruned_open_set

        return []

    def _optimize_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Optimize the path by removing unnecessary zigzags and smoothing the trajectory.
        This function applies a combination of techniques, including removing redundant nodes,
        smoothing sharp turns, and applying Bezier curve interpolation to generate a more
        natural and efficient path.

        Args:
            path (List[Tuple[int, int]]): The original path.

        Returns:
            List[Tuple[int, int]]: The optimized path.
        """
        if len(path) <= 2:
            return path

        # Remove redundant nodes
        optimized_path = [path[0]]
        for i in range(1, len(path) - 1):
            prev_direction = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            next_direction = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            if prev_direction != next_direction:
                optimized_path.append(path[i])
        optimized_path.append(path[-1])

        # Smooth sharp turns
        smoothed_path = [optimized_path[0]]
        for i in range(1, len(optimized_path) - 1):
            prev_node = optimized_path[i - 1]
            current_node = optimized_path[i]
            next_node = optimized_path[i + 1]
            if (
                current_node[0] != prev_node[0]
                and current_node[1] != prev_node[1]
                and current_node[0] != next_node[0]
                and current_node[1] != next_node[1]
            ):
                smoothed_path.append(
                    (
                        (current_node[0] + prev_node[0]) // 2,
                        (current_node[1] + prev_node[1]) // 2,
                    )
                )
                smoothed_path.append(
                    (
                        (current_node[0] + next_node[0]) // 2,
                        (current_node[1] + next_node[1]) // 2,
                    )
                )
            else:
                smoothed_path.append(current_node)
        smoothed_path.append(optimized_path[-1])

        # Apply Bezier curve interpolation for a smoother trajectory
        bezier_path = []
        for i in range(len(smoothed_path) - 1):
            p0 = smoothed_path[i]
            p1 = smoothed_path[i + 1]
            for t in range(BEZIER_RESOLUTION):
                t = t / BEZIER_RESOLUTION
                x = int((1 - t) * p0[0] + t * p1[0])
                y = int((1 - t) * p0[1] + t * p1[1])
                bezier_path.append((x, y))
        bezier_path.append(smoothed_path[-1])

        return bezier_path

    def avoid_collisions(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Check for collisions in the path and dynamically adjust the path to avoid them in real-time.
        This function iterates through the path and checks each node for potential collisions
        with obstacles, boundaries, or the snake's body. If a collision is detected, it
        employs an adaptive A* search to find an optimal alternative path around the collision point,
        considering the current game state and snake's trajectory. The collision avoidance is performed
        iteratively to ensure a collision-free path while maintaining path efficiency and smoothness.

        Args:
            path (List[Tuple[int, int]]): The original path.

        Returns:
            List[Tuple[int, int]]: The collision-free path.
        """
        collision_free_path = path.copy()
        path_length = len(collision_free_path)

        for i in range(path_length):
            node = collision_free_path[i]

            # Check for collisions with obstacles, boundaries, and snake's body
            if (
                node[0] < 0
                or node[0] >= BLOCK_SIZE
                or node[1] < 0
                or node[1] >= BLOCK_SIZE
                or node in self.snake
            ):
                # Perform adaptive A* search to find an alternative path segment
                start = collision_free_path[max(0, i - 1)]
                goal = collision_free_path[min(i + 1, path_length - 1)]

                # Adjust heuristic based on the current game state and snake's trajectory
                adaptive_heuristic = lambda a, b: self.heuristic(a, b, self.direction)

                alternative_path = self.a_star_search(start, goal)

                # Replace the colliding segment with the alternative path
                if alternative_path:
                    collision_free_path[max(0, i - 1) : min(i + 2, path_length)] = (
                        alternative_path
                    )
                else:
                    # If no alternative path found, remove the colliding node
                    collision_free_path.pop(i)
                    path_length -= 1
                    i -= 1

        # Perform a final optimization pass to smooth the path
        optimized_collision_free_path = self._optimize_path(collision_free_path)

        return optimized_collision_free_path

    def get_next_direction(self) -> Tuple[int, int]:
        if not self.path:
            self.path = self.a_star_search(self.snake[0], self.food)
            self.tail_path = self.a_star_search(
                self.food, self.snake[-1], set(self.snake + self.path)
            )

        if len(self.path) > 1:
            next_pos = self.path[1]
            self.path = self.path[1:]
            dx = next_pos[0] - self.snake[0][0]
            dy = next_pos[1] - self.snake[0][1]
            return dx, dy
        elif len(self.tail_path) > 1:
            next_pos = self.tail_path[1]
            self.tail_path = self.tail_path[1:]
            dx = next_pos[0] - self.snake[0][0]
            dy = next_pos[1] - self.snake[0][1]
            return dx, dy
        else:
            self.path = self.a_star_search(self.snake[0], self.food)
            self.tail_path = self.a_star_search(
                self.food, self.snake[-1], set(self.snake + self.path)
            )
            if len(self.path) > 1:
                next_pos = self.path[1]
                self.path = self.path[1:]
                dx = next_pos[0] - self.snake[0][0]
                dy = next_pos[1] - self.snake[0][1]
                return dx, dy

        return self.direction

    def reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        return path[::-1]

    def run(self) -> None:
        running = True
        game_tick = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type is pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            if game_tick % (FPS // GAME_TICK) == 0:
                self.direction = self.get_next_direction()
                game_over = self.play_step()
                if game_over:
                    self.reset_game()

            self.draw_elements()
            self.clock.tick(FPS)
            game_tick += 1

        pygame.quit()


if __name__ == "__main__":
    game = SnakeGameAI()
    game.run()
