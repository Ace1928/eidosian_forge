import pygame
import random
import logging
from heapq import heappush, heappop
from typing import List, Tuple, Dict, Optional

# Initialize logging
logging.basicConfig(
    filename="pygame_snake_game.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Define colors
WHITE, RED, PINK, GREEN, BLUE, BLACK, YELLOW = (
    (255, 255, 255),
    (255, 0, 0),
    (255, 192, 203),
    (0, 255, 0),
    (0, 0, 255),
    (0, 0, 0),
    (255, 255, 0),
)

# Game configuration
CELL_SIZE, GRID_SIZE, FPS, GAME_TICK = 20, 20, 60, 10


class SnakeGameAI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE)
        )
        pygame.display.set_caption("Snake Game AI")
        self.clock = pygame.time.Clock()
        self.reset_game()

    def reset_game(self):
        self.snake = [(GRID_SIZE // 2, GRID_SIZE // 2)]
        self.score = 0
        self.food = None
        self.game_over = False
        self.direction = (1, 0)  # Start moving right
        self.path = []
        self.tail_path = []
        self.place_food()
        logging.info("Game reset.")

    def place_food(self):
        while self.food is None:
            potential_food = (
                random.randint(1, GRID_SIZE - 2),
                random.randint(1, GRID_SIZE - 2),
            )
            if potential_food not in self.snake:
                self.food = potential_food
                logging.debug(f"Food placed at {self.food}")

    def play_step(self):
        if self.game_over:
            return self.game_over
        self.move_snake()
        self.check_collision()
        return self.game_over

    def move_snake(self):
        self.direction = self.get_next_direction()
        next_x, next_y = (
            self.snake[0][0] + self.direction[0],
            self.snake[0][1] + self.direction[1],
        )
        new_head = (next_x, next_y)

        if new_head == self.food:
            self.snake.insert(0, new_head)
            self.score += 1
            self.place_food()
            self.path = self.a_star_search(
                self.snake[0], self.food
            )  # Recalculate path immediately after eating
        elif new_head in self.snake or not (
            1 <= next_x < GRID_SIZE - 1 and 1 <= next_y < GRID_SIZE - 1
        ):
            self.game_over = True
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()

    def check_collision(self):
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

    def draw_elements(self):
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

    def reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
        current: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """
        Reconstruct the path from start to goal using the came_from map filled by the A* search.

        Args:
        came_from (Dict[Tuple[int, int], Optional[Tuple[int, int]]]): The map of each node to its predecessor in the path.
        current (Tuple[int, int]): The current node from which to start reconstructing the path.

        Returns:
        List[Tuple[int, int]]: The reconstructed path as a list of coordinates.
        """
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def get_next_direction(self):
        if not self.path:
            self.path = self.a_star_search(self.snake[0], self.food)

        if self.path:
            next_pos = self.path.pop(0)
            self.direction = (
                next_pos[0] - self.snake[0][0],
                next_pos[1] - self.snake[0][1],
            )
        return self.direction

    def run(self):
        running = True
        game_tick = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            if game_tick % (FPS // GAME_TICK) == 0:
                if not self.game_over:
                    self.play_step()
                else:
                    self.reset_game()  # Reset the game if over

                self.draw_elements()
                self.clock.tick(FPS)
            game_tick += 1

        pygame.quit()


if __name__ == "__main__":
    game = SnakeGameAI()
    game.run()

    def draw_elements(self):
        self.screen.fill(BLACK)
        # Draw borders and elements
        for x, y in self.path + self.tail_path:
            pygame.draw.rect(
                self.screen,
                YELLOW,
                [x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE],
                1,
            )
        for i, (x, y) in enumerate(self.snake):
            color = GREEN if i == 0 else BLUE  # Highlight the head
            pygame.draw.rect(
                self.screen, color, [x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE]
            )
        if self.food:
            pygame.draw.rect(
                self.screen,
                RED,
                [
                    self.food[0] * CELL_SIZE,
                    self.food[1] * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE,
                ],
            )
        pygame.display.update()

    def calculate_path(self):
        # Calculate path dynamically considering current position and the food
        self.path = self.a_star_search(self.snake[0], self.food)

    def run(self):
        running = True
        last_tick = (
            pygame.time.get_ticks()
        )  # Stores the last tick time for game updates

        while running:
            current_time = pygame.time.get_ticks()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            # Check if it's time for the next game update
            if current_time - last_tick > 1000 // GAME_TICK:
                last_tick = current_time  # Update the last_tick to the current time
                if not self.game_over:
                    self.play_step()
                else:
                    self.reset_game()  # Reset the game if over

            self.draw_elements()
            self.clock.tick(
                FPS
            )  # Ensure the program does not run at more than FPS frames per second

        pygame.quit()


if __name__ == "__main__":
    game = SnakeGameAI()
    game.run()
