import pygame as pg
import sys
from random import randint, seed
from collections import deque
from typing import List, Tuple, Deque, Optional, Set, Dict
import logging
import math

# Setup comprehensive logging
logging.basicConfig(
    filename="snake_game.log",
    level=logging.DEBUG,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Constants for game dimensions and block size
SCREEN_WIDTH: int = 600
SCREEN_HEIGHT: int = 400
BLOCK_SIZE: int = 20

# Seed for consistent random behavior
seed(0)

# Initialize pygame
pg.init()
window = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pg.display.set_caption("Dynamic Hamiltonian Snake Game")


class Grid:
    def __init__(self, width: int, height: int, block_size: int) -> None:
        self.width = width
        self.height = height
        self.block_size = block_size
        self.snake_positions: Set[Tuple[int, int]] = set()
        self.fruit_position: Tuple[int, int] = (0, 0)

    def update_snake_position(self, new_positions: Deque[Tuple[int, int]]) -> None:
        self.snake_positions = set(new_positions)

    def update_fruit_position(self, new_position: Tuple[int, int]) -> None:
        self.fruit_position = new_position

    def is_position_occupied(self, position: Tuple[int, int]) -> bool:
        return position in self.snake_positions or position == self.fruit_position


class Fruit:
    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self.color: pg.Color = pg.Color(139, 0, 0)  # Deep red for visibility
        self.radius: int = 10
        self.position: Tuple[int, int] = (0, 0)
        self.relocate()

    def draw(self) -> None:
        pg.draw.circle(
            window,
            self.color,
            (self.position[0] + self.radius, self.position[1] + self.radius),
            self.radius,
        )

    def relocate(self) -> None:
        while True:
            new_x: int = (
                randint(0, (self.grid.width // self.grid.block_size) - 1)
                * self.grid.block_size
            )
            new_y: int = (
                randint(0, (self.grid.height // self.grid.block_size) - 1)
                * self.grid.block_size
            )
            new_position: Tuple[int, int] = (new_x, new_y)
            if not self.grid.is_position_occupied(new_position):
                self.position = new_position
                self.grid.update_fruit_position(new_position)
                break
        logging.info(f"Fruit relocated to {self.position}")


class Snake:
    def __init__(self, grid: Grid, fruit: Fruit) -> None:
        self.grid = grid
        self.fruit = fruit
        self.body: Deque[Tuple[int, int]] = deque([(20, 20), (40, 20), (60, 20)])
        self.growing: int = 0
        self.score: int = 0
        self.grid.update_snake_position(self.body)

    def draw(self) -> None:
        for segment in self.body:
            pg.draw.rect(
                window,
                pg.Color(220, 20, 60),
                pg.Rect(
                    segment[0], segment[1], self.grid.block_size, self.grid.block_size
                ),
            )

    def move(self) -> None:
        """
        Moves the snake according to a calculated path and handles potential collisions.
        """
        next_position: Tuple[int, int] = self.calculate_next_position()
        if next_position in self.body:
            self.restart_game()
        else:
            self.body.appendleft(next_position)
            if self.growing > 0:
                self.growing -= 1
            else:
                self.body.pop()
            self.grid.update_snake_position(self.body)
            logging.info(f"Snake moved to {next_position}")

    def calculate_next_position(self) -> Tuple[int, int]:
        """
        Calculates the next position of the snake using the Theta* pathfinding algorithm.
        """
        path = self.grid.pathfinding.theta_star_path(
            self.body[0], self.fruit.position, self.body, self.grid
        )
        return (
            path[0] if path else self.body[0]
        )  # Continue in the same direction if no path found

    def grow(self) -> None:
        """Increases the size of the snake and updates the score."""
        self.growing += 3
        self.score += 10
        logging.info("Snake grows. Score updated.")

    def restart_game(self) -> None:
        """Restarts the game following a collision or similar event."""
        self.body = deque([(20, 20), (40, 20), (60, 20)])
        self.score = 0
        self.growing = 0
        self.grid.update_snake_position(self.body)
        logging.info("Game restarted.")


class Pathfinding:
    """
    Implements advanced Theta* pathfinding algorithm for the snake to find the most efficient path.
    This algorithm is an enhancement of the A* algorithm, incorporating line-of-sight checks to optimize the path.
    """

    def __init__(self, grid: Grid) -> None:
        self.grid = grid

    def theta_star_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        snake_body: Deque[Tuple[int, int]],
        grid: Grid,
    ) -> List[Tuple[int, int]]:
        """
        Theta* pathfinding algorithm to find the most efficient path from start to goal.
        This method extends A* by integrating line-of-sight checks to potentially skip nodes.

        Args:
            start (Tuple[int, int]): The starting position of the path.
            goal (Tuple[int, int]): The goal position of the path.
            snake_body (Deque[Tuple[int, int]]): The current positions occupied by the snake.
            grid (Grid): The grid on which the pathfinding operates.

        Returns:
            List[Tuple[int, int]]: The path from start to goal as a list of tuples.
        """

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        def line_of_sight(p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
            """Check if a straight line between p1 and p2 is unobstructed."""
            x0, y0 = p1
            x1, y1 = p2
            dx, dy = x1 - x0, y1 - y0
            f = 0
            if dy < 0:
                dy = -dy
                sy = -1
            else:
                sy = 1
            if dx < 0:
                dx = -dx
                sx = -1
            else:
                sx = 1
            if dx >= dy:
                while x0 != x1:
                    f += dy
                    if f >= dx:
                        if grid.is_position_occupied(
                            (x0 + ((sx - 1) // 2), y0 + ((sy - 1) // 2))
                        ):
                            return False
                        y0 += sy
                        f -= dx
                    if f != 0 and grid.is_position_occupied(
                        (x0 + ((sx - 1) // 2), y0 + ((sy - 1) // 2))
                    ):
                        return False
                    if (
                        dy == 0
                        and grid.is_position_occupied((x0 + ((sx - 1) // 2), y0))
                        and grid.is_position_occupied((x0 + ((sx - 1) // 2), y0 - 1))
                    ):
                        return False
                    x0 += sx
            else:
                while y0 != y1:
                    f += dx
                    if f >= dy:
                        if grid.is_position_occupied(
                            (x0 + ((sx - 1) // 2), y0 + ((sy - 1) // 2))
                        ):
                            return False
                        x0 += sx
                        f -= dy
                    if f != 0 and grid.is_position_occupied((x0, y0 + ((sy - 1) // 2))):
                        return False
                    if (
                        dx == 0
                        and grid.is_position_occupied((x0, y0 + ((sy - 1) // 2)))
                        and grid.is_position_occupied((x0 - 1, y0 + ((sy - 1) // 2)))
                    ):
                        return False
                    y0 += sy
            return True

        open_set = PriorityQueue()
        open_set.put((0, start))
        came_from = {start: None}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while not open_set.empty():
            current = open_set.get()[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for dx, dy in [
                (0, grid.block_size),
                (0, -grid.block_size),
                (grid.block_size, 0),
                (-grid.block_size, 0),
            ]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not grid.is_position_occupied(neighbor) or neighbor in snake_body:
                    continue
                tentative_g_score = g_score[current] + grid.block_size

                if neighbor in g_score and tentative_g_score >= g_score[neighbor]:
                    continue

        # Placeholder for pathfinding logic
        return []


class AIController:
    """
    AI Controller that acts as the "player" controlling the snake using pathfinding.
    """

    def __init__(self, snake: Snake, pathfinding: Pathfinding) -> None:
        self.snake = snake
        self.pathfinding = pathfinding

    def make_decision(self) -> None:
        """
        Makes decisions for the snake's next move based on pathfinding.
        """
        self.snake.move()


class Rendering:
    """
    Handles all rendering operations for the game.
    """

    def __init__(self, window, snake: Snake, fruit: Fruit) -> None:
        self.window = window
        self.snake = snake
        self.fruit = fruit

    def render(self) -> None:
        """
        Renders all game objects.
        """
        self.window.fill(pg.Color(0, 0, 0))  # Clear screen with black
        self.snake.draw()
        self.fruit.draw()
        pg.display.flip()  # Update the display


def main():
    """
    Main game loop.
    """
    running = True
    clock = pg.time.Clock()

    grid = Grid(SCREEN_WIDTH, SCREEN_HEIGHT, BLOCK_SIZE)
    fruit = Fruit(grid)
    snake = Snake(grid, fruit)
    pathfinding = Pathfinding(grid)
    ai_controller = AIController(snake, pathfinding)
    renderer = Rendering(window, snake, fruit)

    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False

        ai_controller.make_decision()
        renderer.render()
        clock.tick(10)  # Limit to 10 frames per second

        if snake.score >= 1000:  # Example condition for automatic restart
            snake.restart_game()

    pg.quit()
    sys.exit()


if __name__ == "__main__":
    main()
