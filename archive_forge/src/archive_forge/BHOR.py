# This is the A* pathfinding algorithm
# This works by finding the longest possible path between
# the snake's head and the snake's tail
# The snake will never get trapped because the snake's head
# will always have a way out after reaching the previous tail position
# Apple will be eaten when the snake is on the path

# Rest of the Imports required in alignment with the rest of the classes.
from typing import List, Optional, Tuple
import pygame as pg
from pygame.math import Vector2
import numpy as np
from random import randint

# Initialize Pygame
pg.init()
# Initialize the display
pg.display.init()
# Retrieve the current display information
display_info = pg.display.Info()


# Calculate the block size based on screen resolution to ensure visibility and proportionality
# Define a scaling function for block size relative to screen resolution
def calculate_block_size(screen_width: int, screen_height: int) -> int:
    # Define the reference resolution and corresponding block size
    reference_resolution = (1920, 1080)
    reference_block_size = 20

    # Calculate the scaling factor based on the reference
    scaling_factor_width = screen_width / reference_resolution[0]
    scaling_factor_height = screen_height / reference_resolution[1]
    scaling_factor = min(scaling_factor_width, scaling_factor_height)

    # Calculate the block size dynamically based on the screen size
    dynamic_block_size = max(1, int(reference_block_size * scaling_factor))

    # Ensure the block size does not become too large or too small
    # Set minimum block size to 1x1 pixels and maximum to 30x30 pixels
    adjusted_block_size = min(max(dynamic_block_size, 1), 30)
    return adjusted_block_size


# Apply the calculated block size based on the current screen resolution
block_size = calculate_block_size(display_info.current_w, display_info.current_h)

# Define the border width as equivalent to 3 blocks
border_width = 3 * block_size  # Width of the border to be subtracted from each side

# Define the screen size with a proportional border around the edges
screen_size = (
    display_info.current_w - 2 * border_width,
    display_info.current_h - 2 * border_width,
)

# Define a constant for the border color as solid white
border_color = (255, 255, 255)  # RGB color code for white
clock = pg.time.Clock()
fps = 60
tick_rate = 1000 // fps


# Initial setup of the game environment
def setup_environment() -> Tuple[pg.Surface, pg.time.Clock]:
    """
    Initializes the game environment, setting up the display, and instantiating game objects.
    Returns the screen, snake, apple, search algorithm instance, and the clock for controlling frame rate.
    """
    # Initialize Pygame
    pg.init()
    # Set the screen size using the screen_size constant defined globally
    screen: pg.Surface = pg.display.set_mode(screen_size)
    # Utilize the globally defined clock for controlling the game's frame rate
    clock: pg.time.Clock = clock
    return screen, clock


class Node:
    def __init__(self, x: int, y: int):
        self.position: Vector2 = Vector2(x, y)
        self.parent: Optional["Node"] = None
        self.f: float = 0.0
        self.g: float = 0.0
        self.h: float = 0.0

    def is_equal(self, other: "Node") -> bool:
        return self.position == other.position


class SearchAlgorithm:
    def __init__(self, snake, apple):
        self.snake = snake
        self.apple = apple

    def refresh_maze(self):
        maze = []
        for i in range(20):
            row = []
            for j in range(40):
                row.append(0)
            maze.append(row)
        for i in range(len(self.snake.body)):
            # Ensure that the indices are integers
            maze[int(self.snake.body[i].y)][int(self.snake.body[i].x)] = -1
        head_position = self.snake.get_head_position()
        tail_position = self.snake.get_tail_position()
        maze[int(head_position.y)][int(head_position.x)] = 1
        maze[int(tail_position.y)][int(tail_position.x)] = 2
        return maze

    def get_path(self):
        maze = self.refresh_maze()
        start, end = None, None
        for i in range(40):
            for j in range(20):
                if maze[j][i] == 1:
                    start = {"x": i, "y": j}
                elif maze[j][i] == 2:
                    end = {"x": i, "y": j}
        node_path = self.astar(maze, start, end)
        vector_path = []
        for node in node_path:
            vector_path.append(Vector2(node.position.x, node.position.y))
        self.snake.path = vector_path

    def astar(self, maze, start, end):
        start_node = Node(start["x"], start["y"])
        end_node = Node(end["x"], end["y"])
        open_list = []
        closed_list = []
        open_list.append(start_node)
        possible_paths = []
        adjacent_squares = [
            [0, -1],
            [0, 1],
            [-1, 0],
            [1, 0],
        ]
        while len(open_list) > 0:
            current_node = open_list[0]
            current_index = 0
            index = 0
            for i in range(len(open_list)):
                if open_list[i].f > current_node.f:
                    current_node = open_list[i]
                    current_index = index
                index += 1
            open_list.pop(current_index)
            closed_list.append(current_node)
            if current_node.is_equal(end_node):
                path = []
                current = current_node
                while current is not None:
                    path.append(current)
                    current = current.parent
                possible_paths.append(list(reversed(path)))
            children = []
            for i in range(len(adjacent_squares)):
                node_position = [
                    int(current_node.position.x + adjacent_squares[i][0]),
                    int(current_node.position.y + adjacent_squares[i][1]),
                ]
                if 0 <= node_position[0] <= 39:
                    if 0 <= node_position[1] <= 19:
                        if maze[node_position[1]][node_position[0]] != -1:
                            new_node = Node(node_position[0], node_position[1])
                            children.append(new_node)
            for i in range(len(children)):
                if_in_closed_list = False
                for j in range(len(closed_list)):
                    if children[i].is_equal(closed_list[j]):
                        if_in_closed_list = True
                if not if_in_closed_list:
                    children[i].g = current_node.g + 2
                    children[i].h = abs(
                        children[i].position.x - end_node.position.x
                    ) + abs(children[i].position.y - end_node.position.y)
                    children[i].f = children[i].g + children[i].h
                    present = False
                    for j in range(len(open_list)):
                        if (
                            children[i].is_equal(open_list[j])
                            and children[i].g < open_list[j].g
                        ):
                            present = True
                        elif (
                            children[i].is_equal(open_list[j])
                            and children[i].g >= open_list[j].g
                        ):
                            open_list[j] = children[i]
                            open_list[j].parent = current_node
                    if not present:
                        children[i].parent = current_node
                        open_list.append(children[i])
        path = []
        for i in range(len(possible_paths)):
            if len(possible_paths[i]) > len(path):
                path = possible_paths[i]
        return path

        import time
        import pygame as pg
        from typing import List, Set, Tuple
        from random import randint

        # Constants for the maze simulation
        MAZE_WIDTH = 40
        MAZE_HEIGHT = 20
        NUM_SCENARIOS = 10
        SCENARIO_DURATION = 5  # seconds
        OBSTACLE_MIN = 50
        OBSTACLE_MAX = 150
        SCREEN_MARGIN = 0.1  # 10% margin on each side

        # Colors
        COLOR_BACKGROUND = (255, 255, 255)  # White
        COLOR_OBSTACLE = (0, 0, 0)  # Black
        COLOR_START = (0, 255, 0)  # Green
        COLOR_GOAL = (255, 0, 0)  # Red
        COLOR_PATH = (0, 0, 255)  # Blue

        def generate_complex_maze(
            width: int, height: int, num_obstacles: int
        ) -> List[List[int]]:
            """
            Generate a complex maze with specified dimensions and number of obstacles.

            Args:
                width (int): The width of the maze.
                height (int): The height of the maze.
                num_obstacles (int): The number of obstacles to place in the maze.

            Returns:
                List[List[int]]: A 2D list representing the maze where -1 indicates an obstacle.
            """
            maze = [[0 for _ in range(width)] for _ in range(height)]
            obstacles: Set[Tuple[int, int]] = set()
            while len(obstacles) < num_obstacles:
                x, y = randint(0, width - 1), randint(0, height - 1)
                if maze[y][x] == 0:
                    maze[y][x] = -1
                    obstacles.add((x, y))
            return maze

        def draw_maze(screen: pg.Surface, maze: List[List[int]], block_size: int):
            """
            Draw the maze on the Pygame screen.

            Args:
                screen (pg.Surface): The Pygame screen to draw on.
                maze (List[List[int]]): The maze to draw.
                block_size (int): The size of each block in pixels.
            """
            for y, row in enumerate(maze):
                for x, cell in enumerate(row):
                    rect = pg.Rect(
                        x * block_size, y * block_size, block_size, block_size
                    )
                    if cell == -1:
                        pg.draw.rect(screen, COLOR_OBSTACLE, rect)
                    else:
                        pg.draw.rect(screen, COLOR_BACKGROUND, rect)

        def draw_path(screen: pg.Surface, path: List[Node], block_size: int):
            """
            Draw the path on the Pygame screen.

            Args:
                screen (pg.Surface): The Pygame screen to draw on.
                path (List[Node]): The path to draw.
                block_size (int): The size of each block in pixels.
            """
            for node in path:
                rect = pg.Rect(
                    node.position.x * block_size,
                    node.position.y * block_size,
                    block_size,
                    block_size,
                )
                pg.draw.rect(screen, COLOR_PATH, rect)

        def draw_start_goal(
            screen: pg.Surface,
            start: Tuple[int, int],
            goal: Tuple[int, int],
            block_size: int,
        ):
            """
            Draw the start and goal positions on the Pygame screen.

            Args:
                screen (pg.Surface): The Pygame screen to draw on.
                start (Tuple[int, int]): The starting position (x, y).
                goal (Tuple[int, int]): The goal position (x, y).
                block_size (int): The size of each block in pixels.
            """
            start_rect = pg.Rect(
                start[0] * block_size, start[1] * block_size, block_size, block_size
            )
            goal_rect = pg.Rect(
                goal[0] * block_size, goal[1] * block_size, block_size, block_size
            )
            pg.draw.rect(screen, COLOR_START, start_rect)
            pg.draw.rect(screen, COLOR_GOAL, goal_rect)

        def run_scenario(
            screen: pg.Surface,
            start: Tuple[int, int],
            goal: Tuple[int, int],
            maze: List[List[int]],
            block_size: int,
        ):
            """
            Run a single scenario of pathfinding from start to goal in the given maze.

            Args:
                screen (pg.Surface): The Pygame screen to draw on.
                start (Tuple[int, int]): The starting position (x, y).
                goal (Tuple[int, int]): The goal position (x, y).
                maze (List[List[int]]): The maze to navigate.
                block_size (int): The size of each block in pixels.
            """
            start_node = Node(start[0], start[1])
            end_node = Node(goal[0], goal[1])
            path = astar(
                maze, {"x": start[0], "y": start[1]}, {"x": goal[0], "y": goal[1]}
            )

            draw_maze(screen, maze, block_size)
            draw_start_goal(screen, start, goal, block_size)
            pg.display.flip()
            time.sleep(1)  # Pause for a moment to view the maze

            if path:
                draw_path(screen, path, block_size)
                pg.display.flip()
                time.sleep(1)  # Pause for a moment to view the path

                for node in path:
                    rect = pg.Rect(
                        node.position.x * block_size,
                        node.position.y * block_size,
                        block_size,
                        block_size,
                    )
                    pg.draw.rect(screen, COLOR_PATH, rect)
                    pg.display.flip()
                    time.sleep(0.1)  # Delay between each step of the path
            else:
                print(f"No path found from {start} to {goal}")

        def main():
            """
            Main function to run multiple scenarios of pathfinding in complex mazes using Pygame.
            """
            pg.init()
            screen_width, screen_height = 800, 600
            screen = pg.display.set_mode((screen_width, screen_height))
            pg.display.set_caption("Maze Pathfinding Simulation")
            clock = pg.time.Clock()

            block_size_x = (screen_width * (1 - 2 * SCREEN_MARGIN)) // MAZE_WIDTH
            block_size_y = (screen_height * (1 - 2 * SCREEN_MARGIN)) // MAZE_HEIGHT
            block_size = min(block_size_x, block_size_y)

            for scenario in range(NUM_SCENARIOS):
                num_obstacles = randint(OBSTACLE_MIN, OBSTACLE_MAX)
                maze = generate_complex_maze(MAZE_WIDTH, MAZE_HEIGHT, num_obstacles)
                start = (randint(0, MAZE_WIDTH - 1), randint(0, MAZE_HEIGHT - 1))
                goal = (randint(0, MAZE_WIDTH - 1), randint(0, MAZE_HEIGHT - 1))

                print(f"Scenario {scenario + 1}:")
                run_scenario(screen, start, goal, maze, block_size)
                time.sleep(SCENARIO_DURATION)

                for event in pg.event.get():
                    if event.type == pg.QUIT or (
                        event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE
                    ):
                        pg.quit()
                        return

                screen.fill(COLOR_BACKGROUND)
                clock.tick(60)

            pg.quit()

        if __name__ == "__main__":
            main()
