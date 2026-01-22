import pygame
import random
import heapq
import logging
from collections import defaultdict
from typing import List, Tuple, Dict, Set


def create_grid(size: int) -> List[List[int]]:
    """
    Constructs a two-dimensional square grid of specified size where each cell within the grid is initialized to zero. This grid is represented as a list of lists, where each inner list corresponds to a row in the grid.

    Args:
        size (int): The dimension of the grid, specifically indicating both the number of rows and the number of columns, as the grid is square.

    Returns:
        List[List[int]]: A 2D list where each element within the sublist (representing a row) is initialized to zero. The outer list encapsulates all these rows, thus forming the complete grid structure.
    """
    return [[0 for _ in range(size)] for _ in range(size)]


def convert_to_graph(
    grid: List[List[int]],
) -> Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]:
    """
    Converts a two-dimensional grid into a graph representation where each cell in the grid is treated as a node in the graph. The nodes are connected to their adjacent nodes (up, down, left, right) with edges that have randomly assigned weights. This conversion facilitates the representation of the grid in a format that is amenable to graph-theoretic algorithms.

    Args:
        grid (List[List[int]]): A two-dimensional list of integers representing the grid. Each element in the grid is assumed to be an integer.

    Returns:
        Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]: A dictionary representing the graph. The keys are tuples representing the coordinates of each node in the grid. The values are lists of tuples, where each tuple contains a tuple representing the coordinates of an adjacent node and a float representing the weight of the edge connecting the node to the adjacent node.
    """
    size = len(grid)
    graph = {}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(size):
        for j in range(size):
            connections = []
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < size and 0 <= nj < size:
                    connections.append(((ni, nj), random.random()))
            graph[(i, j)] = connections

    return graph


def prim_mst(
    graph: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    start_vertex: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """
    Implements Prim's algorithm to compute the Minimum Spanning Tree (MST) from a given graph starting from a specified vertex. Prim's algorithm is a greedy algorithm that finds a minimum spanning tree for a weighted undirected graph. This means it finds a subset of the edges that forms a tree that includes every vertex, where the total weight of all the edges in the tree is minimized.

    Args:
        graph (Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]): This is a dictionary representing the graph. The keys are tuples representing the coordinates of each node in the grid, and the values are lists of tuples, where each tuple contains a tuple representing the coordinates of an adjacent node and a float representing the weight of the edge connecting the node to the adjacent node.
        start_vertex (Tuple[int, int]): This tuple represents the starting vertex from which the MST will begin to be computed.

    Returns:
        List[Tuple[int, int]]: This list contains tuples representing the nodes that form the Minimum Spanning Tree, starting from the start_vertex and expanding out to include all reachable nodes in the graph in a way that minimizes the total edge weight.
    """
    visited = set([start_vertex])
    edges = [(weight, start_vertex, to) for to, weight in graph[start_vertex]]
    heapq.heapify(edges)
    mst = [start_vertex]

    while edges:
        weight, frm, to = heapq.heappop(edges)
        if to not in visited:
            visited.add(to)
            mst.append(to)
            for neighbor, weight in graph[to]:
                if neighbor not in visited:
                    heapq.heappush(edges, (weight, to, neighbor))

    return mst


def draw_path(
    screen,
    path: List[Tuple[int, int]],
    cell_size: int,
    current_index: int,
    max_length: int,
    frame_count: int,
):
    """
    Draw the path on the screen using Pygame with a gradient neon glow effect that smoothly transitions through a spectrum of colors. Each segment of the path also changes its color dynamically, creating a gradient of gradients effect. Additionally, implement a fading glow effect for the segments.

    Args:
        screen: Pygame screen object.
        path (List[Tuple[int, int]]): The path to draw.
        cell_size (int): Size of each cell in the grid.
        current_index (int): Current index in the path for the animation.
        max_length (int): Maximum number of segments to display at once.
        frame_count (int): Current frame count to adjust the color dynamically.
    """
    start = max(0, current_index - max_length)
    end = current_index + 1
    segments = path[start:end]

    def compute_color(index, frame_count):
        base_colors = [
            (0, 0, 0),
            (255, 0, 0),
            (255, 165, 0),
            (255, 255, 0),
            (0, 128, 0),
            (0, 0, 255),
            (75, 0, 130),
            (238, 130, 238),
            (255, 255, 255),
            (128, 128, 128),
            (0, 0, 0),
        ]
        num_colors = len(base_colors)
        base_index = index % num_colors
        next_index = (base_index + 1) % num_colors
        ratio = (index % num_colors) / num_colors
        r = int(
            base_colors[base_index][0] * (1 - ratio)
            + base_colors[next_index][0] * ratio
        )
        g = int(
            base_colors[base_index][1] * (1 - ratio)
            + base_colors[next_index][1] * ratio
        )
        b = int(
            base_colors[base_index][2] * (1 - ratio)
            + base_colors[next_index][2] * ratio
        )
        r = (r + 2 * frame_count) % 256
        g = (g + 2 * frame_count) % 256
        b = (b + 2 * frame_count) % 256
        return (r, g, b)

    for i, (x, y) in enumerate(segments):
        color = compute_color(i + frame_count, frame_count)
        rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, [min(c + 50, 255) for c in color], rect, 1)
        if i < len(segments) - 1:
            fade_color = [max(c - 10 * (len(segments) - i), 0) for c in color]
            pygame.draw.rect(screen, fade_color, rect, 1)


def main():
    """
    The main function initializes the pygame window, sets up the game grid and graph, and handles the main game loop.
    """
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    pygame.init()
    grid_size = 100
    cell_size = 5
    screen = pygame.display.set_mode(
        (grid_size * cell_size, grid_size * cell_size), pygame.RESIZABLE
    )
    pygame.display.set_caption("Dynamic Hamiltonian Cycle Visualization")

    logging.info("Creating grid and converting to graph.")
    grid = create_grid(grid_size)
    graph = convert_to_graph(grid)
    current_path = prim_mst(graph, (0, 0))
    path_index = 0
    path_length = 1000

    running = True
    clock = pygame.time.Clock()
    while running:
        frame_count = pygame.time.get_ticks() // 10
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                logging.info("Quitting the game.")

        screen.fill((0, 0, 0))
        draw_path(screen, current_path, cell_size, path_index, path_length, frame_count)
        pygame.display.flip()
        path_index += 1

        if path_index >= len(current_path) - path_length // 2:
            new_start_vertex = current_path[-1]
            logging.debug(f"Recalculating path from vertex {new_start_vertex}.")
            current_path = current_path[:path_index] + prim_mst(graph, new_start_vertex)

        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
